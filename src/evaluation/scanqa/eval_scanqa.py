import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch.multiprocessing as mp

sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from src.evaluation.utils.common_utils import (
    chunk_dataset,
    flatten,
    prepare_spatial_mllm_inputs,
    save_json,
    setup_logging,
)


# JJ: import from common_utils
from src.evaluation.utils.common_utils import load_model_and_processor


def build_user_message(item: Dict, annotation_dir: Path, video_folder: Path, video_nframes: int, sample_fps: float = None) -> Dict:
    """Create the chat-style message payload for a single sample."""
    # build question
    video_file_name = item["video"].split("/")[-1] + ".mp4"
    video_file_path = video_folder / video_file_name

    video_content = {
        "type": "video",
        "video": str(video_file_path),
        "resized_height": 480,
        "resized_width": 640,
    }
    
    # JJ: Sanity check - nframes and sample_fps should not both be provided
    assert (video_nframes is None) or (sample_fps is None), \
        f"Cannot specify both nframes ({video_nframes}) and sample_fps ({sample_fps}). Use one or the other."
    
    # JJ: Set do_sample_frames if neither is provided
    video_content["do_sample_frames"] = (video_nframes is None) and (sample_fps is None)
    
    # JJ: Set nframes or fps if provided
    if video_nframes is not None:
        video_content["nframes"] = video_nframes
    if sample_fps is not None:
        video_content["fps"] = sample_fps

    return {
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": item["conversations"][0]["value"]},
        ],
    }


def prepare_chat_batch(
    batch_data: List[Dict],
    processor: Any,
    model_type: str,
    annotation_dir: Path,
    video_folder: Path,
    video_nframes: int,
    sample_fps: float = None,
) -> Tuple[Dict, List[str]]:
    """Prepare batch for inference: build prompts, process video, and tokenize."""
    batch_messages = [[build_user_message(item, annotation_dir, video_folder, video_nframes, sample_fps)] for item in batch_data]
    prompts_text = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in batch_messages
    ]
    prompts_text_copy = prompts_text.copy()

    video_inputs = []
    image_inputs = []
    for example in batch_messages:
        images, videos = process_vision_info(example)
        if images:
            image_inputs.extend(images)
        elif videos:
            video_inputs.extend(videos)
        else:
            raise ValueError("Each example must contain either images or videos.")

    print(prompts_text)
    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    return batch, prompts_text_copy


def inference_batch(batch_inputs: Dict, model: Any, processor: Any) -> List[str]:
    """Run inference on the batch inputs."""
    batch_inputs.to(model.device)
    if "image_tchw" in batch_inputs and batch_inputs["image_tchw"] is not None:
        batch_inputs["image_tchw"] = [image_tchw_i.to(model.device) for image_tchw_i in batch_inputs["image_tchw"]]
    if "video_tchw" in batch_inputs and batch_inputs["video_tchw"] is not None:
        batch_inputs["video_tchw"] = [video_tchw_i.to(model.device) for video_tchw_i in batch_inputs["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs, **generation_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def postprocess_batch(batch_data: List[Dict], batch_output_text: List[str], prompts_text: List[str]) -> List[Dict]:
    """Post-process outputs: clean text, calculate rewards, and structure results."""
    batch_results = []
    for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
        batch_results.append(
            {
                "dataset": sample["metadata"]["dataset"],
                "sample_id": sample["id"],
                "prompt": prompt,
                "pred_response": model_output.strip(),
                "gt_response": sample["conversations"][1]["value"],
                "question_type": sample["metadata"]["question_type"],
            }
        )

    return batch_results


def evaluate_scanqa_sqa3d(
    data_chunk, model_type, model_path, batch_size, annotation_path, video_folder, output_path, video_nframes, sample_fps=None, use_visual=None, use_geo=None
):
    setup_logging()
    # JJ: load model and processor with customized use_visual/use_geo
    model, processor = load_model_and_processor(model_type, model_path, use_visual=use_visual, use_geo=use_geo)
    final_output = []

    for i in tqdm(range(0, len(data_chunk), batch_size), desc="Evaluating ScanQA/SQA3D"):
        batch_data = data_chunk[i : i + batch_size]
        batch_llm_inputs, prompts_text = prepare_chat_batch(
            batch_data, processor, model_type, annotation_path, video_folder, video_nframes, sample_fps
        )
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # Checkpoint partial results every 10 batches or at the end
        if (i + 1) % 10 == 0 or (i + 1) == len(data_chunk):
            save_json(output_path, final_output)

    return final_output


def run_worker(
    gpu_id, data_chunk, model_type, model_path, batch_size, annotation_path, video_folder, output_path, video_nframes, sample_fps=None, use_visual=None, use_geo=None
):
    """Worker function to run evaluation on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    evaluate_scanqa_sqa3d(
        data_chunk, model_type, model_path, batch_size, annotation_path, video_folder, output_path, video_nframes, sample_fps, use_visual, use_geo
    )


def main(args):
    setup_logging()

    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir).resolve() / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = Path(args.annotation_path).resolve()
    video_folder = Path(args.video_folder).resolve()

    # Load Data
    with open(str(annotation_path)) as f:
        data = json.load(f)
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("Evaluation requires at least one CUDA device.")

    print(f"Starting evaluation on {n_gpu} GPUs...")

    # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]

    processes = []
    output_paths = []

    for idx, data_chunk in enumerate(chunk_dataset(data, n_gpu)):
        output_path_gpu = output_dir / f"results_{args.model_type}_{idx}.json"
        output_paths.append(output_path_gpu)

        # Select GPU ID
        gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else str(idx)

        p = mp.Process(
            target=run_worker,
            args=(
                gpu_id,
                data_chunk,
                args.model_type,
                args.model_path,
                args.batch_size,
                annotation_path,
                video_folder,
                output_path_gpu,
                args.nframes,
                args.sample_fps,
                args.use_visual,
                args.use_geo,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_output = []
    for path in output_paths:
        if path.exists():
            with open(path, "r") as f:
                final_output.extend(json.load(f))
        else:
            print(f"Warning: Output file {path} not found.")

    save_json(
        output_dir / f"results_{args.model_type}.json",
        final_output,
    )
    print(f"Finished evaluation for scanqa/sqa3d.")


if __name__ == "__main__":
    def int_or_none(x):
        if x.lower() == "none":
            return None
        return int(x)

    parser = argparse.ArgumentParser(description="Evaluate model on VSIBench dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--model_type", type=str, default="spatial-mllm", help="Type of the model.")
    # parser.add_argument("--nframes", type=int, default=32, help="Number of frames to sample from each video.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (forced to 1).")
    parser.add_argument("--annotation_path", type=str, required=True, help="Scanqa or SQA3D annotation file path.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing videos.")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
    parser.add_argument(
        "--output_name", type=str, default="eval_vsibench", help="Directory to save evaluation results."
    )
    # JJ to support parse None
    parser.add_argument(
        "--nframes",
        type=int_or_none,
        default=None,
        help="Number of frames to sample from each video (or 'None' for no limit)."
    )  
    # JJ: video sampling config
    parser.add_argument("--sample_fps", type=float, default=None, help="Sample FPS for video (default: None, use nframes)")
    # JJ: connector config
    parser.add_argument("--use_visual", type=lambda x: x.lower() == 'true', default=None, help="Use visual embeddings (true/false, default: None for model default)")
    parser.add_argument("--use_geo", type=lambda x: x.lower() == 'true', default=None, help="Use geo embeddings (true/false, default: None for model default)")
    args = parser.parse_args()

    main(args)
