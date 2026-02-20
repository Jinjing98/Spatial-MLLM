import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image

# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vl_utils import process_vision_info
# JJ: import from common_utils
from src.evaluation.utils.common_utils import load_model_and_processor, prepare_spatial_mllm_inputs, prepare_spatial_mllm_inputs_with_framesid
from src.evaluation.utils.common_utils import construct_msg_qwen2_5_vl, construct_msg_qwen3_vl
from src.evaluation.utils.common_utils import elaborate_batch_info_debug


# JJ: Helper function to load selected_frames from json if exists
def load_selected_frames_from_path(video_path: str):
    """
    Load selected_frames from selected_frames.json if it exists in the video_path directory.
    
    Args:
        video_path: Path to video file or directory containing frames
        
    Returns:
        List[int] or None: Selected frame indices if json file exists, None otherwise
    """
    # If video_path is a directory, look for selected_frames.json
    if os.path.isdir(video_path):
        json_path = os.path.join(video_path, "selected_frames.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                selected_frames = data.get("selected_frames", None)
                if selected_frames is not None:
                    print(f"[INFO] Loaded selected_frames from {json_path}: {selected_frames}")
                    return selected_frames
    return None


def main(
    video_path: str = "assets/arkitscenes_41069025.mp4",
    text: str = "How many chair(s) are in this room?\nPlease answer the question using a single word or phrase.",  # na question
    # text: str = "Measuring from the closest point of each object, what is the distance between the sofa and the stove (in meters)?\nPlease answer the question using a single word or phrase.",  # na question
    # text: str = "If I am standing by the stove and facing the tv, is the sofa to my front-left, front-right, back-left, or back-right?\nThe directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).Options:\nA. back-left\nB. front-right\nC. back-right\nD. front-left\nAnswer with the option's letter from the given choices directly.",  # mca question
    model_type: str = "spatial-mllm",
    model_path: str = "checkpoints/Spatial-MLLM-v1.1-Instruct-135K",
    # JJ extend for flexiblity.
    mp4_nframes: int | None = 16,  # JJ: Number of frames to sample from mp4 video (None to use sample_fps or auto)
    sample_fps: float | None = None,  # JJ: Sample FPS for video (.mp4 will auto-detect original fps)
    raw_fps: float | None = None,  # JJ: Original FPS (only for pre-sampled image folders with qwen3-vl)
    use_visual: bool | None = None,  # JJ: Use visual embeddings (None for model default)
    use_geo: bool | None = None,  # JJ: Use geo embeddings (None for model default)
):
    print(f"[Inference] model_type={model_type}, mp4_nframes={mp4_nframes}, sample_fps={sample_fps}")
    torch.cuda.empty_cache()

    # load the model
    # JJ: load model and processor with customized use_visual/use_geo
    model, processor = load_model_and_processor(model_type, model_path, use_visual=use_visual, use_geo=use_geo)

    # JJ: Handle both video file and image folder - use appropriate message constructor
    if model_type == "qwen3-vl":
        messages = construct_msg_qwen3_vl(text, video_path, mp4_nframes=mp4_nframes, sample_fps=sample_fps, raw_fps=raw_fps)
    else:
        messages = construct_msg_qwen2_5_vl(text, video_path, mp4_nframes=mp4_nframes, sample_fps=sample_fps)
    # Preparation for inference
    prompts_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # JJ: Extract vision info from messages
    # For Qwen3-VL, we need to get video metadata from fetch_video
    if model_type == "qwen3-vl":
        from qwen_vl_utils import extract_vision_info
        from src.evaluation.utils.common_utils import gen_videos_metadata
        
        vision_infos = extract_vision_info(messages)
        image_inputs, video_inputs, videos_metadata = gen_videos_metadata(vision_infos)

        batch = processor(
            text=[prompts_text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=videos_metadata,
            do_sample_frames=False,  # JJ: Video already sampled by fetch_video; don't re-sample
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
    else:
        # For other models, use standard process_vision_info
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        batch = processor(
            text=[prompts_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

    # JJ: elaborate the batch info for debug
    elaborate_batch_info_debug(processor, batch)
    # JJ elaborate the update
    if model_type in ["spatial-mllm"]:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)
    elif model_type in ["custom-spatial-mllm"]:
        # JJ: Load selected_frames if available
        selected_frames = load_selected_frames_from_path(video_path)
        selected_frames_list = [selected_frames] if selected_frames is not None else None
        # JJ: Use prepare_spatial_mllm_inputs_with_framesid for custom-spatial-mllm to support mRoPE_readaptT
        batch = prepare_spatial_mllm_inputs_with_framesid(batch, video_inputs, image_inputs, selected_frames_list)
    elif model_type in ["qwen2.5-vl"]:
        pass # Remain as it is
    elif model_type in ["qwen3-vl"]:
        pass # Remain as it is
        # raise NotImplementedError("Qwen3 VL is not implemented yet.")
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    batch.to(model.device)
    if "image_tchw" in batch and batch["image_tchw"] is not None:
        batch["image_tchw"] = [image_tchw_i.to(model.device) for image_tchw_i in batch["image_tchw"]]
    if "video_tchw" in batch and batch["video_tchw"] is not None:
        batch["video_tchw"] = [video_tchw_i.to(model.device) for video_tchw_i in batch["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    # Start time measurement
    time_0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**batch,**generation_kwargs)
    time_taken = time.time() - time_0

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
    ]
    num_generated_tokens = sum(len(ids) for ids in generated_ids_trimmed)

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"Time taken for inference: {time_taken:.2f} seconds")
    print(f"GPU Memory taken for inference: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Number of generated tokens: {num_generated_tokens}")
    print(f"Time taken per token: {time_taken / num_generated_tokens:.4f} seconds/token")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    tyro.cli(main, description="Run inference.")
