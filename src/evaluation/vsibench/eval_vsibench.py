import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch.multiprocessing as mp

sys.path.append(str(Path(__file__).resolve().parents[3]))

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from datasets import load_dataset
from src.evaluation.utils.common_utils import (
    chunk_dataset,
    flatten,
    gen_videos_metadata,
    prepare_spatial_mllm_inputs,
    prepare_spatial_mllm_inputs_with_framesid,
    save_json,
    setup_logging,
)
from src.evaluation.vsibench.dataset_utils import MCA_QUESTION_TYPES, NA_QUESTION_TYPES, clean_text, vsi_reward


# Constants
SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "mca": "Answer with the option's letter from the given choices directly.",
    "na": "Please answer the question using a single word or phrase.",
}

# JJ: import from common_utils
from src.evaluation.utils.common_utils import load_model_and_processor


def build_user_message(item: Dict, video_dir: Path, video_nframes: int, sample_fps: float = None) -> Tuple[Dict, Optional[List[int]]]:
    """Create the chat-style message payload for a single sample."""
    # build question
    raw_question = SFT_QUESTION_TEMPLATE.format(Question=item["question"])
    q_type = item["question_type"]
    if q_type in MCA_QUESTION_TYPES:
        options = item.get("options") or []
        if not options:
            raise ValueError("Multiple-choice samples must include 'options'.")
        options_text = "Options:\n" + "\n".join(options)
        question = f"{raw_question}\n{options_text}\n{SFT_TYPE_TEMPLATE['mca']}"
    elif q_type in NA_QUESTION_TYPES:
        question = f"{raw_question}\n{SFT_TYPE_TEMPLATE['na']}"
    else:
        raise ValueError(f"Unknown question type: {q_type}")

    text_content = {"type": "text", "text": question}
    video_content = {"type": "video"}
    selected_frames = None

    if (video_dir / item["dataset"] / (item["scene_name"] + ".mp4")).exists():  # mp4 video file
        video_path = (video_dir / item["dataset"] / (item["scene_name"] + ".mp4")).resolve()
        video_content["video"] = str(video_path)
        
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
    elif (video_dir / item["dataset"] / item["scene_name"]).exists():  # folder of frames
        frame_folder = (video_dir / item["dataset"] / item["scene_name"]).resolve()
        video_path = sorted(glob.glob(str(frame_folder / "*.png")))
        assert (
            len(video_path) == video_nframes
        ), f"Number of frames in {frame_folder} ({len(video_path)}) does not match expected {video_nframes}."
        video_content["video"] = video_path
        
        # JJ: Load selected_frames if available
        selected_frames_json = frame_folder / "selected_frames.json"
        if selected_frames_json.exists():
            with open(selected_frames_json, 'r') as f:
                metadata = json.load(f)
                selected_frames = metadata.get("selected_frames")
    else:
        raise FileNotFoundError(
            f"Data file not found for video_dir" f"{video_dir}, dataset {item['dataset']}, scene {item['scene_name']}"
        )

    return {
        "role": "user",
        "content": [video_content, text_content],
    }, selected_frames


def prepare_chat_batch(
    batch_data: List[Dict],
    processor: Any,
    model_type: str,
    video_dir: Path,
    video_nframes: int,
    sample_fps: float = None,
) -> Tuple[Dict, List[str]]:
    """Prepare batch for inference: build prompts, process video, and tokenize."""
    batch_messages_and_frames = [build_user_message(item, video_dir, video_nframes, sample_fps) for item in batch_data]
    batch_messages = [[msg] for msg, _ in batch_messages_and_frames]
    batch_selected_frames = [frames for _, frames in batch_messages_and_frames]# JJ

    prompts_text = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in batch_messages
    ]
    prompts_text_copy = prompts_text.copy()

    # JJ : Split vision processing by model type (qwen3-vl needs extract_vision_info + gen_videos_metadata)
    if model_type in ["qwen3-vl", "spatial-mllm-qwen3"]:
        from qwen_vl_utils import extract_vision_info
        video_inputs = []
        image_inputs = []
        videos_metadata_all = []
        for example in batch_messages:
            vision_infos = extract_vision_info(example)
            imgs, vids, vids_meta = gen_videos_metadata(vision_infos)
            if imgs:
                image_inputs.extend(imgs)
            if vids:
                video_inputs.extend(vids)
            if vids_meta:
                videos_metadata_all.extend(vids_meta)

        batch = processor(
            text=prompts_text,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            video_metadata=videos_metadata_all if videos_metadata_all else None,
            do_sample_frames=False,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
    else:
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

        batch = processor(
            text=prompts_text,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

    if "spatial-mllm" == model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)
    elif "custom-spatial-mllm" == model_type:
        # JJ
        batch = prepare_spatial_mllm_inputs_with_framesid(batch, video_inputs, image_inputs, batch_selected_frames)
    elif "spatial-mllm-qwen3" == model_type:
        # JJ : spatial-mllm-qwen3 needs tchw for pose computation
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)
    elif model_type in ["qwen2.5-vl", "qwen3-vl"]:
        pass  # JJ : No special batch preparation needed
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return batch, prompts_text_copy


def inference_batch(batch_inputs: Dict, model: Any, processor: Any) -> List[str]:
    """Run inference on the batch inputs."""
    batch_inputs.to(model.device)
    if "image_tchw" in batch_inputs and batch_inputs["image_tchw"] is not None:
        batch_inputs["image_tchw"] = [image_tchw_i.to(model.device) for image_tchw_i in batch_inputs["image_tchw"]]
    if "video_tchw" in batch_inputs and batch_inputs["video_tchw"] is not None:
        batch_inputs["video_tchw"] = [video_tchw_i.to(model.device) for video_tchw_i in batch_inputs["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs,**generation_kwargs)

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def postprocess_batch(
    batch_data: List[Dict], batch_output_text: List[str], prompts_text: List[str]
) -> List[Dict]:
    """Post-process outputs: clean text, calculate rewards, and structure results."""
    batch_results = []
    for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
        clean_ans = clean_text(model_output)
        clean_ans_gt = clean_text(sample.get("ground_truth", ""))
        reward = vsi_reward(clean_ans_gt, clean_ans, sample["question_type"])

        batch_results.append(
            {
                "sample": sample,
                "prompt": prompt,
                "model_output": model_output,
                "cleaned_model_output": clean_ans,
                "cleaned_gt_answer": clean_ans_gt,
                "reward": reward,
                "correct": reward == 1.0,
            }
        )

    return batch_results


def calculate_metrics(results):
    """Calculate detailed metrics (per-type scores/counts, micro/macro)."""
    if not results:
        return {
            "per_question_type": {},
            "acc": {"micro": 0.0, "macro": 0.0},
            "mra": {"micro": 0.0, "macro": 0.0},
            "all": {"micro": 0.0, "macro": 0.0},
            "prune_ratio": {"mean": None},
        }

    df = pd.DataFrame(
        [
            {
                "reward": res.get("reward", 0.0),
                "question_type": res["sample"].get("question_type"),
            }
            for res in results
        ]
    )
    df["is_na"] = df["question_type"].isin(NA_QUESTION_TYPES)

    def safe_mean(series):
        return float(series.mean()) if len(series) else 0.0

    # Per-question-type scores and counts
    per_qtype = {
        qtype: {"score": float(group["reward"].mean()), "count": int(len(group))}
        for qtype, group in df.groupby("question_type")
    }

    # Micro scores
    acc_mask = ~df["is_na"]
    mra_mask = df["is_na"]
    micro_acc = safe_mean(df.loc[acc_mask, "reward"])
    micro_mra = safe_mean(df.loc[mra_mask, "reward"])
    micro_all = safe_mean(df["reward"])

    # Macro scores (average of per-type scores)
    acc_qtypes = [q for q in per_qtype if q not in NA_QUESTION_TYPES]
    mra_qtypes = [q for q in per_qtype if q in NA_QUESTION_TYPES]

    macro_acc = safe_mean(pd.Series([per_qtype[q]["score"] for q in acc_qtypes]))
    macro_mra = safe_mean(pd.Series([per_qtype[q]["score"] for q in mra_qtypes]))
    macro_all = safe_mean(pd.Series([v["score"] for v in per_qtype.values()]))

    return {
        "per_question_type": per_qtype,
        "acc": {"micro": micro_acc, "macro": macro_acc},
        "mra": {"micro": micro_mra, "macro": macro_mra},
        "all": {"micro": micro_all, "macro": macro_all},
    }


# JJ : LaTeX formatting functions for terminal output
def format_latex_row_style1(metrics: Dict, model_name: str) -> str:
    """Format metrics to LaTeX row - Style 1: simple with RelDir avg. Values are multiplied by 100 and shown as X.X format."""
    per_qtype = metrics.get("per_question_type", {})
    
    def get_score(qtype: str) -> str:
        return f"{per_qtype[qtype]['score'] * 100:.1f}" if qtype in per_qtype else "-"
    
    # Compute RelDir weighted average
    reldir_types = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]
    scores, counts = [], []
    for qt in reldir_types:
        if qt in per_qtype:
            scores.append(per_qtype[qt]["score"])
            counts.append(per_qtype[qt]["count"])
    reldir_avg = sum(s * c for s, c in zip(scores, counts)) / sum(counts) if counts and sum(counts) > 0 else 0.0
    
    all_micro = metrics.get("all", {}).get("micro", 0.0)
    
    values = [
        model_name,
        get_score("object_counting"),
        get_score("object_abs_distance"),
        get_score("object_size_estimation"),
        get_score("room_size_estimation"),
        get_score("object_rel_distance"),
        f"{reldir_avg * 100:.1f}",
        get_score("route_planning"),
        get_score("obj_appearance_order"),
        f"{all_micro * 100:.1f}"
    ]
    return " & ".join(values) + " \\\\"


def format_latex_row_style2(metrics: Dict, model_name: str) -> str:
    """Format metrics to LaTeX row - Style 2: RelDir as easy/mid/hard in one cell. Values are multiplied by 100 and shown as X.X format."""
    per_qtype = metrics.get("per_question_type", {})
    
    def get_score(qtype: str) -> str:
        return f"{per_qtype[qtype]['score'] * 100:.1f}" if qtype in per_qtype else "-"
    
    # Get RelDir easy/mid/hard scores
    reldir_easy = per_qtype.get("object_rel_direction_easy", {}).get("score", 0.0) * 100
    reldir_mid = per_qtype.get("object_rel_direction_medium", {}).get("score", 0.0) * 100
    reldir_hard = per_qtype.get("object_rel_direction_hard", {}).get("score", 0.0) * 100
    reldir_combined = f"{reldir_easy:.1f}/{reldir_mid:.1f}/{reldir_hard:.1f}"
    
    mra_micro = metrics.get("mra", {}).get("micro", 0.0)
    acc_micro = metrics.get("acc", {}).get("micro", 0.0)
    all_micro = metrics.get("all", {}).get("micro", 0.0)
    
    values = [
        model_name,
        get_score("object_counting"),
        get_score("object_abs_distance"),
        get_score("object_size_estimation"),
        get_score("room_size_estimation"),
        get_score("object_rel_distance"),
        reldir_combined,  # easy/mid/hard in one cell
        get_score("route_planning"),
        get_score("obj_appearance_order"),
        f"{mra_micro * 100:.1f}",
        f"{acc_micro * 100:.1f}",
        f"{all_micro * 100:.1f}"
    ]
    return " & ".join(values) + " \\\\"


def print_latex_results(metrics: Dict, model_name: str, dataset_metrics: Dict[str, Dict] = None):
    """Print LaTeX formatted results to terminal."""
    print("\n" + "="*80)
    print("📋 LaTeX Formatted Results")
    print("="*80)
    
    # Style 1
    print("\n[Style 1: Simple with RelDir average]")
    print("Model & ObjCnt & AbsDist & ObjSz & RoomSz & RelDist & RelDir & RoutePl & ApprOrd & Overall \\\\")
    print("\\hline")
    print(format_latex_row_style1(metrics, model_name))
    
    # Style 2
    print("\n[Style 2: RelDir as E/M/H in one cell + MRA + ACC]")
    print("Model & ObjCnt & AbsDist & ObjSz & RoomSz & RelDist & RelDir & RoutePl & ApprOrd & MRA & ACC & Overall \\\\")
    print("\\hline")
    print(format_latex_row_style2(metrics, model_name))
    
    # Per-dataset rows (Style 1)
    if dataset_metrics:
        print("\n[Per-Dataset Results (Style 1)]")
        print("Model & ObjCnt & AbsDist & ObjSz & RoomSz & RelDist & RelDir & RoutePl & ApprOrd & Overall \\\\")
        print("\\hline")
        for dataset_name in sorted(dataset_metrics.keys()):
            row = format_latex_row_style1(dataset_metrics[dataset_name], f"{model_name}-{dataset_name}")
            print(row)
    
    print("="*80 + "\n")


def evaluate_vsibench(vsi_data, model_type, model_path, batch_size, video_dir, output_path, video_nframes, sample_fps=None, use_visual=None, use_geo=None, use_pose_rope=False, pose_enc_type="PTHW", mrope_section=None):
    """Evaluate model on a specific dataset. Forces batch size to 1."""

    setup_logging()
    # JJ: load model and processor with customized use_visual/use_geo
    model, processor = load_model_and_processor(model_type, model_path, use_visual=use_visual, use_geo=use_geo)
    
    # 🆕 NEW: Apply Pose RoPE monkey patch for custom-spatial-mllm (Qwen2.5)
    if model_type == "custom-spatial-mllm" and use_pose_rope:
        from src.custom_qwenvl.model.custom_spatial_mllm_pose_rope import patch_model_with_pose_rope
        
        # Print user-level configuration before patching
        print(f"[Evaluation] 🔧 Applying Pose RoPE configuration:")
        print(f"[Evaluation]    - pose_enc_type: {pose_enc_type}")
        print(f"[Evaluation]    - mrope_section: {mrope_section if mrope_section else 'default (will be determined by pose_enc_type)'}")
        
        # 🆕 NEW: Validate config consistency if model was trained with Pose RoPE
        if hasattr(model.config, 'pose_rope_config'):
            saved_config = model.config.pose_rope_config
            print(f"[Evaluation] 📋 Detected saved Pose RoPE config in checkpoint:")
            print(f"[Evaluation]    - use_pose_rope: {saved_config.get('use_pose_rope')}")
            print(f"[Evaluation]    - pose_enc_type: {saved_config.get('pose_enc_type')}")
            print(f"[Evaluation]    - mrope_section: {saved_config.get('mrope_section')}")
            
            # Warn if mismatch
            if not saved_config.get('use_pose_rope', False):
                print(f"[Evaluation] ⚠️  WARNING: Checkpoint was trained WITHOUT Pose RoPE, but you're enabling it now!")
                print(f"[Evaluation]    This may cause unexpected behavior. Consider using --use_pose_rope=False")
            
            if saved_config.get('pose_enc_type') != pose_enc_type:
                print(f"[Evaluation] ⚠️  WARNING: pose_enc_type mismatch!")
                print(f"[Evaluation]    Checkpoint: {saved_config.get('pose_enc_type')}")
                print(f"[Evaluation]    Current: {pose_enc_type}")
        
        model = patch_model_with_pose_rope(
            model,
            use_pose_rope=True,
            pose_enc_type=pose_enc_type,
            mrope_section=mrope_section,  # 🆕 NEW: Pass custom mrope_section if provided
            # Note: All Temporal & Pose parameters are inherited from model.__init__
        )
        
        # Dynamic message based on actual pose_enc_type
        if pose_enc_type == "PTHW":
            dims_desc = "4D Pose-aware RoPE (P+T+H+W)"
        elif pose_enc_type == "PHW":
            dims_desc = "3D Pose-aware RoPE (P+H+W, ignore temporal)"
        elif pose_enc_type == "THW":
            dims_desc = "3D standard mRoPE (T+H+W, ignore pose)"
        else:
            dims_desc = f"RoPE with pose_enc_type={pose_enc_type}"
        
        print(f"[Evaluation] ✅ Monkey patch applied: Model now uses {dims_desc}")
        # Print final configuration after patching
        actual_mrope_section = model.config.rope_scaling.get("mrope_section", "not found")
        print(f"[Evaluation] 📊 Final mrope_section: {actual_mrope_section}")
    elif model_type == "custom-spatial-mllm" and not use_pose_rope:
        # 🆕 NEW: Warn if checkpoint expects Pose RoPE but we're not using it
        if hasattr(model.config, 'pose_rope_config') and model.config.pose_rope_config.get('use_pose_rope', False):
            print(f"[Evaluation] ⚠️  WARNING: Checkpoint was trained WITH Pose RoPE, but you're NOT enabling it!")
            print(f"[Evaluation]    Checkpoint config: {model.config.pose_rope_config}")
            print(f"[Evaluation]    Consider using --use_pose_rope to match training configuration")
        print(f"[Evaluation] ℹ️  Using standard 3D mRoPE (T+H+W)")
    
    # 🆕 NEW: Apply Pose RoPE monkey patch for spatial-mllm-qwen3
    elif model_type == "spatial-mllm-qwen3" and use_pose_rope:
        assert pose_enc_type == "PHW", "Qwen3 only supports PHW pose encoding type"
        
        from src.custom_qwen3vl.model.spatial_mllm_qwen3_pose_rope import patch_qwen3_with_pose_rope
        
        print(f"[Evaluation] 🔧 Applying Qwen3 Pose RoPE configuration:")
        print(f"[Evaluation]    - pose_enc_type: {pose_enc_type}")
        print(f"[Evaluation]    - mrope_section: {mrope_section if mrope_section else 'default [24, 20, 20]'}")
        
        # Validate config consistency
        if hasattr(model.config, 'pose_rope_config'):
            saved_config = model.config.pose_rope_config
            print(f"[Evaluation] 📋 Detected saved Pose RoPE config:")
            print(f"[Evaluation]    - pose_enc_type: {saved_config.get('pose_enc_type')}")
            print(f"[Evaluation]    - mrope_section: {saved_config.get('mrope_section')}")
        
        model = patch_qwen3_with_pose_rope(
            model,
            use_pose_rope=True,
            pose_enc_type=pose_enc_type,
            mrope_section=mrope_section,
        )
        
        actual_mrope_section = model.config.text_config.rope_parameters.get("mrope_section", "not found")
        print(f"[Evaluation] 📊 Final mrope_section: {actual_mrope_section}")
    elif model_type == "spatial-mllm-qwen3" and not use_pose_rope:
        print(f"[Evaluation] ℹ️  Using Qwen3 standard 3D mRoPE (T+H+W)")
    
    final_output = []

    for i in tqdm(range(0, len(vsi_data), batch_size), desc="Evaluating VSIBench"):
        batch_data = vsi_data[i : i + batch_size]
        batch_llm_inputs, prompts_text = prepare_chat_batch(batch_data, processor, model_type, video_dir, video_nframes, sample_fps)
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # Checkpoint partial results every 10 batches or at the end
        if (i + 1) % 10 == 0 or (i + 1) == len(vsi_data):
            save_json(output_path, final_output)

    return final_output


def run_worker(gpu_id, vsi_data, model_type, model_path, batch_size, video_dir, output_path, video_nframes, sample_fps=None, use_visual=None, use_geo=None, use_pose_rope=False, pose_enc_type="PTHW", mrope_section=None):
    """Worker function to run evaluation on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    evaluate_vsibench(vsi_data, model_type, model_path, batch_size, video_dir, output_path, video_nframes, sample_fps, use_visual, use_geo, use_pose_rope, pose_enc_type, mrope_section)


def main(args):
    setup_logging()

    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir).resolve() / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = Path(args.annotation_dir).resolve()
    if args.video_dir:
        video_dir = Path(args.video_dir).resolve()
    else:
        video_dir = annotation_dir

    vsi_data = load_dataset(str(annotation_dir), "full")["test"]
    
    # Filter by datasets if specified
    if args.datasets:
        print(f"Filtering dataset to datasets: {args.datasets}")
        vsi_data = vsi_data.filter(lambda x: x["dataset"] in args.datasets)
        print(f"Filtered dataset size: {len(vsi_data)}")
    
    # Filter by question types if specified
    if args.question_types:
        print(f"Filtering dataset to question types: {args.question_types}")
        vsi_data = vsi_data.filter(lambda x: x["question_type"] in args.question_types)
        print(f"Filtered dataset size: {len(vsi_data)}")
    
    # Filter by scene names if specified
    if args.scene_names:
        print(f"Filtering dataset to scene names: {args.scene_names}")
        vsi_data = vsi_data.filter(lambda x: x["scene_name"] in args.scene_names)
        print(f"Filtered dataset size: {len(vsi_data)}")
    
    # JJ : Eval phase (skippable via --skip_eval)
    if not args.skip_eval:
        n_gpu = torch.cuda.device_count()
        if n_gpu <= 0:
            raise RuntimeError("VSIBench evaluation requires at least one CUDA device.")

        print(f"Starting evaluation on {n_gpu} GPUs...")

        # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
        else:
            gpu_ids = [str(i) for i in range(n_gpu)]
        print('GPU IDs: ', gpu_ids)
        processes = []
        output_paths = []

        for idx, data_chunk in enumerate(chunk_dataset(vsi_data, n_gpu)):
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
                    video_dir,
                    output_path_gpu,
                    args.nframes,
                    args.sample_fps,
                    args.use_visual,
                    args.use_geo,
                    args.use_pose_rope,
                    args.pose_enc_type,
                    args.mrope_section,  # 🆕 NEW: Pass custom mrope_section
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("Skipping evaluation phase (--skip_eval).")

    # JJ : Metrics phase (skippable via --skip_metric)
    if not args.skip_metric:
        # JJ : determine input directory for reading results (only use input_dir when --skip_eval)
        if args.input_dir and args.skip_eval:
            input_dir = Path(args.input_dir).resolve()
            print(f"Reading existing results from: {input_dir}")
        elif args.input_dir and not args.skip_eval:
            print(f"Warning: --input_dir is ignored when not using --skip_eval. Reading from output_dir.")
            input_dir = output_dir
        else:
            input_dir = output_dir
            print(f"Reading results from output directory: {input_dir}")
        
        # Load results from existing shard files
        merged_results_path = input_dir / f"results_{args.model_type}.json"
        shard_paths = sorted(input_dir.glob(f"results_{args.model_type}_*.json"))

        final_output = []
        if shard_paths:
            for path in shard_paths:
                with open(path, "r") as f:
                    final_output.extend(json.load(f))
        elif merged_results_path.exists():
            with open(merged_results_path, "r") as f:
                final_output = json.load(f)
        else:
            print(f"Warning: No result files found in {input_dir} for model_type={args.model_type}.")
        
        # JJ : raise error if no results found to prevent misleading all-zero metrics
        if not final_output:
            raise FileNotFoundError(
                f"No result files found in {input_dir} for model_type={args.model_type}. "
                f"Cannot compute metrics without evaluation results. "
                f"Either run without --skip_eval to generate results, or check the input directory."
            )

        # JJ : filter results by datasets, question_types, scene_names (always apply for consistency)
        original_count = len(final_output)
        if args.datasets:
            final_output = [res for res in final_output if res["sample"].get("dataset") in args.datasets]
            print(f"Filtered by datasets {args.datasets}: {original_count} -> {len(final_output)} samples")
        
        if args.question_types:
            original_count = len(final_output)
            final_output = [res for res in final_output if res["sample"].get("question_type") in args.question_types]
            print(f"Filtered by question_types {args.question_types}: {original_count} -> {len(final_output)} samples")
        
        if args.scene_names:
            original_count = len(final_output)
            final_output = [res for res in final_output if res["sample"].get("scene_name") in args.scene_names]
            print(f"Filtered by scene_names {args.scene_names}: {original_count} -> {len(final_output)} samples")
        
        # JJ : if no samples after filtering, save empty metrics and proceed
        if not final_output:
            print(f"⚠️  Warning: No samples remain after filtering. Saving empty metrics.")
            empty_metrics = calculate_metrics([])
            save_json(merged_results_path, [])
            save_json(output_dir / f"metrics_{args.model_type}.json", empty_metrics)
            print(f"Finished evaluation for vsibench (no samples).")
            return

        # JJ : re-compute reward with latest vsi_reward logic so --skip_eval picks up fixes
        for res in final_output:
            clean_pred = clean_text(res.get("model_output", ""))
            clean_gt = clean_text(res.get("sample", {}).get("ground_truth", ""))
            qtype = res.get("sample", {}).get("question_type", "")
            res["cleaned_model_output"] = clean_pred
            res["cleaned_gt_answer"] = clean_gt
            res["reward"] = vsi_reward(clean_gt, clean_pred, qtype)
            res["correct"] = res["reward"] == 1.0

        # Compute the overall metrics across shards.
        final_acc_dict = calculate_metrics(final_output)
        save_json(
            merged_results_path,
            final_output,
        )
        save_json(
            output_dir / f"metrics_{args.model_type}.json",
            final_acc_dict,
        )
        print(f"Finished evaluation for vsibench.")
        print(f"Final Metrics (Overall): {final_acc_dict}")
        
        # JJ : compute and save per-dataset metrics automatically
        dataset_metrics_dict = {}
        unique_datasets = sorted(set(res["sample"].get("dataset") for res in final_output if res["sample"].get("dataset")))
        if unique_datasets:
            print(f"\n📦 Computing per-dataset metrics for: {unique_datasets}")
            for dataset in unique_datasets:
                dataset_results = [res for res in final_output if res["sample"].get("dataset") == dataset]
                if not dataset_results:
                    # Skip empty dataset (should not happen, but defensive)
                    print(f"  ⚠️  {dataset}: 0 samples, skipping")
                    continue
                dataset_metrics = calculate_metrics(dataset_results)
                dataset_metrics_dict[dataset] = dataset_metrics
                dataset_metrics_path = output_dir / f"metrics_{args.model_type}_{dataset}.json"
                save_json(dataset_metrics_path, dataset_metrics)
                print(f"  ✓ {dataset}: {len(dataset_results)} samples -> {dataset_metrics_path.name}")
                print(f"    ACC: micro={dataset_metrics['acc']['micro']:.4f}, macro={dataset_metrics['acc']['macro']:.4f}")
                print(f"    MRA: micro={dataset_metrics['mra']['micro']:.4f}, macro={dataset_metrics['mra']['macro']:.4f}")
        else:
            print(f"\n📦 No dataset information found in results.")
        
        # JJ : print LaTeX formatted results for easy copy-paste
        print_latex_results(final_acc_dict, args.model_type, dataset_metrics_dict if dataset_metrics_dict else None)

    else:
        print("Skipping metrics computation (--skip_metric).")


if __name__ == "__main__":
    def int_or_none(x):
        if x.lower() == "none":
            return None
        return int(x)

    parser = argparse.ArgumentParser(description="Evaluate model on VSIBench dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--model_type", type=str, default="spatial-mllm", help="Type of the model.")
    # parser.add_argument("--nframes", type=int, default=16, help="Number of frames to sample from each video.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (forced to 1).")
    parser.add_argument(
        "--annotation_dir", type=str, required=True, help="Directory containing the VSIBench data files."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing the video frame files, if none, use annotation_dir.",
    )
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
    parser.add_argument(
        "--output_name", type=str, default="eval_vsibench", help="Directory to save evaluation results."
    )
    # JJ : input_dir for --skip_eval to read from a different directory
    parser.add_argument(
        "--input_dir", type=str, default=None, 
        help="Directory to read existing results from (used with --skip_eval). If not specified, defaults to output_dir/output_name."
    )
    parser.add_argument(
        "--question_types",
        type=str,
        nargs="+",
        default=None,
        help="List of question types to evaluate. If not specified, all question types will be evaluated.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of datasets to evaluate (e.g., arkitscenes, scannet, scannetpp). If not specified, all datasets will be evaluated.",
    )
    parser.add_argument(
        "--scene_names",
        type=str,
        nargs="+",
        default=None,
        help="List of scene names to evaluate. If not specified, all scenes will be evaluated.",
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
    # JJ: 4D Pose RoPE config
    parser.add_argument("--use_pose_rope", action="store_true", default=False, help="Enable 4D Pose-aware RoPE (P+T+H+W) instead of 3D mRoPE (T+H+W)")
    parser.add_argument("--pose_enc_type", type=str, default="PTHW", help="Pose encoding type ('PTHW', 'PHW', or 'THW')")
    parser.add_argument("--mrope_section", type=int, nargs='+', default=None, help="Custom mrope_section (e.g., 16 24 24 for 3D or 8 8 24 24 for 4D)")
    # JJ : skip flags for eval / metric phases
    parser.add_argument("--skip_eval", action="store_true", default=False, help="Skip the evaluation (inference) phase, only compute metrics from existing results.")
    parser.add_argument("--skip_metric", action="store_true", default=False, help="Skip the metrics computation phase, only run evaluation.")
    args = parser.parse_args()

    main(args)
