import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

import glob


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# JJ: alter connector config for evaluation
def eval_altering_connector_config(config, use_visual=None, use_geo=None):
    """Alter connector config during inference."""
    if use_visual is not None:
        config.connector_config["use_visual"] = use_visual
        print(f"[INFO] Set use_visual={use_visual}")
    if use_geo is not None:
        config.connector_config["use_geo"] = use_geo
        print(f"[INFO] Set use_geo={use_geo}")
    return config

# JJ: unified model loader
def load_model_and_processor(model_type: str, model_path: str, use_visual=None, use_geo=None):
    """Load model and processor. Optionally alter use_visual/use_geo for spatial-mllm models."""
    if model_type == "custom-spatial-mllm":
        # JJ Custom modules
        from src.custom_qwenvl.model.custom_spatial_mllm import CustomSpatialMLLMConfig, CustomSpatialMLLMForConditionalGeneration
        from transformers import Qwen2_5_VLProcessor

        config = CustomSpatialMLLMConfig.from_pretrained(model_path)
        # JJ: alter connector config if specified
        config = eval_altering_connector_config(config, use_visual=use_visual, use_geo=use_geo)
        model = CustomSpatialMLLMForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=True)
        hf_processor_path='Diankun/Spatial-MLLM-v1.1-Instruct-135K' # JJ: need when eval our sft model
        processor = Qwen2_5_VLProcessor.from_pretrained(hf_processor_path, use_fast=True)

        return model, processor
    
    elif model_type == "spatial-mllm":
        from transformers import Qwen2_5_VLProcessor

        from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration

        config = SpatialMLLMConfig.from_pretrained(model_path)
        # JJ: alter connector config if specified
        config = eval_altering_connector_config(config, use_visual=use_visual, use_geo=use_geo)
        model = SpatialMLLMForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=True)
        hf_processor_path='Diankun/Spatial-MLLM-v1.1-Instruct-135K' # JJ: need when eval our sft model
        processor = Qwen2_5_VLProcessor.from_pretrained(hf_processor_path, use_fast=True)
        return model, processor

    elif model_type == "qwen2.5-vl":
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=True)
        return model, processor

    elif model_type == "qwen3-vl":
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        # processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        processor = Qwen3VLProcessor.from_pretrained(model_path, use_fast=True)
        return model, processor

    raise ValueError(f"Unknown model type: {model_type}")

# JJ: flexiable fps num_frames gen.
def construct_msg_qwen2_5_vl(text, video_path, mp4_nframes=None, sample_fps=None, raw_fps=None):
    video_path_obj = Path(video_path)
    
    if video_path_obj.is_file():  # Video file (.mp4, etc.)
        video_content = {
            "type": "video",
            "video": video_path,
            "do_sample_frames": (mp4_nframes is None) and (sample_fps is None), # JJ
            # "nframes": mp4_nframes,
        }

        assert (mp4_nframes is None) or (sample_fps is None), "Either mp4_nframes or sample_fps must be provided, not both"
        # JJ: set nframes or fps if provided
        if mp4_nframes is not None:
            video_content["nframes"] = mp4_nframes
        if sample_fps is not None:
            video_content["fps"] = sample_fps

    elif video_path_obj.is_dir():  # Image folder (pretend as video)
        image_files = sorted(glob.glob(str(video_path_obj / "*.png")))
        if not image_files:
            raise FileNotFoundError(f"No PNG files found in {video_path}")
        
        video_content = {
            "type": "video",
            "video": image_files,  # List of image paths
            # JJ: refer to common_utils.py where I copied source code: raw_fps, sample_fps
            # Note: Do NOT set nframes for image list
        }
    else:
        raise FileNotFoundError(f"Path not found: {video_path}")

    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]
    return messages

# JJ: flexiable fps num_frames gen for Qwen3-VL
def construct_msg_qwen3_vl(text, video_path, mp4_nframes=None, sample_fps=None, raw_fps=None):
    video_path_obj = Path(video_path)
    
    if video_path_obj.is_file():  # Video file (.mp4, etc.)
        # JJ: Sanity check - nframes and sample_fps should not both be provided
        assert (mp4_nframes is None) or (sample_fps is None), \
            "Either mp4_nframes or sample_fps must be provided, not both"
        
        video_content = {
            "type": "video",
            "video": video_path,
        }
        
        # JJ: For Qwen3-VL, set sampling parameters
        # Note: For .mp4 files, decord will auto-detect fps from video metadata
        if sample_fps is not None:
            video_content["fps"] = sample_fps  # Target sampling rate
        elif mp4_nframes is not None:
            video_content["nframes"] = mp4_nframes
        else:
            # Auto sampling mode
            video_content["do_sample_frames"] = True
            
    elif video_path_obj.is_dir():  # Image folder (pretend as video)
        image_files = sorted(glob.glob(str(video_path_obj / "*.png")))
        if not image_files:
            raise FileNotFoundError(f"No PNG files found in {video_path}")
        
        video_content = {
            "type": "video",
            "video": image_files,  # List of image paths
        }
        
        # JJ: For pre-sampled frames in directory, optionally provide fps metadata for qwen3-vl
        # Note: directory already contains sampled frames, so nframes/fps params are ignored for frame count
        if sample_fps is not None and raw_fps is not None:
            # Provide fps info for timestamp calculation (used by fetch_video when processing frame list)
            video_content["sample_fps"] = sample_fps  # Used in fetch_video line 43
            video_content["raw_fps"] = raw_fps  # Used in fetch_video line 45
        # Note: Do NOT set nframes for image list (already determined by folder contents)
        
    else:
        raise FileNotFoundError(f"Path not found: {video_path}")

    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]
    return messages

def gen_videos_metadata(vision_infos):
    '''
    Generate videos metadata from vision infos. Used for Qwen3-VL.
    '''
    from qwen_vl_utils import fetch_image, fetch_video 
    image_inputs = []
    video_inputs = []
    videos_metadata = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            # JJ: fetch_video with both flags returns: ((video_tensor, raw_metadata_dict), sample_fps)
            (video, raw_meta), _sample_fps = fetch_video(
                vision_info, 
                return_video_sample_fps=True, 
                return_video_metadata=True
            )
            video_inputs.append(video)
            # JJ: Pass full metadata including frames_indices (needed by text processor
            # for timestamp calculation). We'll set do_sample_frames=False so the
            # video processor won't try to re-index the already-sampled tensor.
            videos_metadata.append({
                "fps": raw_meta.get("fps"),
                "total_num_frames": raw_meta.get("total_num_frames"),
                "frames_indices": raw_meta.get("frames_indices"),
            })
            print(f"[Debug] video: {video.shape[0]} sampled frames, original_fps={raw_meta.get('fps')}, total_original_frames={raw_meta.get('total_num_frames')}, num_indices={len(raw_meta.get('frames_indices', []))}")

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
        videos_metadata = None
    
    return image_inputs, video_inputs, videos_metadata
    
def prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs):
    """
        Prepare inputs for Spatial MLLM model.
        Batch: Dict return by the processor
        video_input and image_inputs is returned by process_vision_info
        
        video_inputs: List[torch.Tensor[Int]] | List[torch.Tensor[Float]] | List[List[PIL.Image]]
        image_inputs: List[PIL.Image]
    """
    video_tchw = []
    image_tchw = []

    if video_inputs:
        for video_input in video_inputs:
            if isinstance(video_input, torch.Tensor):
                video_input = video_input.float() / 255.0  # Normalize to [0, 1]
            elif isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
                # Convert list of PIL Images to tensor
                video_input = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]).float() / 255.0
            else:
                raise ValueError("Unsupported video input format.")
            video_tchw.append(video_input)
    
    if image_inputs:
        for image_input in image_inputs:
            if isinstance(image_input, Image.Image):
                image_input = torch.tensor(np.array(image_input)).permute(2, 0, 1).float() / 255.0
            else:
                raise ValueError("Unsupported image input format.")
            image_tchw.append(image_input)


    print("--------------------------------")
    print('Updated batch keys in spatial mllm batch:')
    print(f"video_tchw[0] shape: {video_tchw[0].shape}")
    print(f"image_tchw: {image_tchw}")
    print("--------------------------------")

    # Extend with , ignore pixel_values_videos
    batch.update({
        "video_tchw": video_tchw if video_tchw else None,
        "image_tchw": image_tchw if image_tchw else None,
    })


    return batch

# JJ
def prepare_spatial_mllm_inputs_with_framesid(batch, video_inputs, image_inputs, selected_frames_list=None):
    """
        Prepare inputs for Spatial MLLM model.
        Batch: Dict return by the processor
        video_input and image_inputs is returned by process_vision_info
        
        video_inputs: List[torch.Tensor[Int]] | List[torch.Tensor[Float]] | List[List[PIL.Image]]
        image_inputs: List[PIL.Image]
        selected_frames_list: List[Optional[List[int]]]
    """
    video_tchw = []
    image_tchw = []

    if video_inputs:
        for video_input in video_inputs:
            if isinstance(video_input, torch.Tensor):
                video_input = video_input.float() / 255.0  # Normalize to [0, 1]
            elif isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
                # Convert list of PIL Images to tensor
                video_input = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]).float() / 255.0
            else:
                raise ValueError("Unsupported video input format.")
            video_tchw.append(video_input)
    
    if image_inputs:
        for image_input in image_inputs:
            if isinstance(image_input, Image.Image):
                image_input = torch.tensor(np.array(image_input)).permute(2, 0, 1).float() / 255.0
            else:
                raise ValueError("Unsupported image input format.")
            image_tchw.append(image_input)

    batch.update({
        "video_tchw": video_tchw if video_tchw else None,
        "image_tchw": image_tchw if image_tchw else None,
    })
    
    # Add selected_frames if provided
    if selected_frames_list is not None and any(frames is not None for frames in selected_frames_list):
        # For batch processing, we take the first non-None selected_frames
        # (assuming batch size is 1 or all items in batch share the same scene)
        selected_frames = next((frames for frames in selected_frames_list if frames is not None), None)
        if selected_frames is not None:
            batch["selected_frames"] = selected_frames

    return batch
 

# Elaborate the update. JJ
def elaborate_batch_info_debug(processor, batch):
    from transformers import Qwen2_5_VLProcessor
    if isinstance(processor, Qwen2_5_VLProcessor):
        print("--------------------------------")
        print(f"Video_tchw in original Qwen {type(processor).__name__} processor inputs:")
        print('existing batch keys:', list(batch.keys()))
        print(f"second_per_grid_t:{batch['second_per_grid_ts']}")
        print(f"video_grid_thw: {batch['video_grid_thw']}")
        print(f"pixel_values_videos[0] shape: {batch['pixel_values_videos'][0].shape}")
        print("--------------------------------")
        return

    from transformers import Qwen3VLProcessor # avialable in transformers 5.x
    if isinstance(processor, Qwen3VLProcessor):
        print("--------------------------------")
        print(f"Video_tchw in original Qwen {type(processor).__name__} processor inputs:")
        print('existing batch keys:', list(batch.keys()))
        print(f"no second_per_grid_t in Qwen3 vl processor inputs")
        print(f"video_grid_thw: {batch['video_grid_thw']}")
        print(f"pixel_values_videos[0] shape: {batch['pixel_values_videos'][0].shape}")
        print("--------------------------------")
        return

    raise ValueError(f"Invalid processor type: {type(processor)}")

def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Split dataset into roughly equal shards."""
    if num_shards <= 0:
        return [dataset]

    chunk_size = math.ceil(len(dataset) / num_shards)
    return [
        [dataset[i] for i in range(start, min(start + chunk_size, len(dataset)))]
        for start in range(0, len(dataset), chunk_size)
    ]

def flatten(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in nested for item in sublist]

def save_json(output_path: str | Path, data: Any):
    """Save data to json file."""
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")

def save_jsonl(output_path: str | Path, data: List[Any]):
    """Save list of data to jsonl file."""
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")

# # JJ: copy from source code for reference
# from typing import Union, Optional, Tuple, Dict, Any
# from qwen_vl_utils import extract_vision_info, fetch_image, fetch_video
# def process_vision_info_official_qwen3_vl(
#     conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
#     return_video_kwargs: bool = False,
#     return_video_metadata: bool = False,
#     image_patch_size: int = 14,
# ) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:

#     vision_infos = extract_vision_info(conversations)
#     ## Read images or videos
#     image_inputs = []
#     video_inputs = []
#     video_sample_fps_list = []
#     for vision_info in vision_infos:
#         if "image" in vision_info or "image_url" in vision_info:
#             image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
#         elif "video" in vision_info:
#             video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True,
#                         image_patch_size=image_patch_size, return_video_metadata=return_video_metadata)
#             video_sample_fps_list.append(video_sample_fps)
#             video_inputs.append(video_input)
#         else:
#             raise ValueError("image, image_url or video should in content.")
#     if len(image_inputs) == 0:
#         image_inputs = None
#     if len(video_inputs) == 0:
#         video_inputs = None

#     video_kwargs = {'do_sample_frames': False}
#     if not return_video_metadata: # BC for qwen2.5vl
#         video_kwargs.update({'fps': video_sample_fps_list})

#     if return_video_kwargs:
#         return image_inputs, video_inputs, video_kwargs
#     return image_inputs, video_inputs

# # JJ: copy from source code for reference
# from qwen_vl_utils import smart_resize, get_video_reader_backend, VIDEO_READER_BACKENDS, VIDEO_MIN_TOKEN_NUM, VIDEO_MAX_TOKEN_NUM, MAX_NUM_WORKERS_FETCH_VIDEO
# from qwen_vl_utils import SPATIAL_MERGE_SIZE, FRAME_FACTOR, ceil_by_factor, MODEL_SEQ_LEN
# from torchvision import io, transforms
# from torchvision.transforms import InterpolationMode
# from concurrent.futures import ThreadPoolExecutor
# import logging
# logger = logging.getLogger(__name__)

# def fetch_video_official_qwen3_vl(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
#                 return_video_metadata: bool = False) -> Union[torch.Tensor, List[Image.Image]]:
#     image_factor = image_patch_size * SPATIAL_MERGE_SIZE
#     VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
#     VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
#     if isinstance(ele["video"], str):
#         video_reader_backend = get_video_reader_backend()
#         try:
#             video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
#         except Exception as e:
#             logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
#             video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
#     else:
#         # The input is a list of frames
#         assert isinstance(ele["video"], (list, tuple))
#         process_info = ele.copy()
#         process_info.pop("type", None)
#         process_info.pop("video", None)
#         # use ThreadPoolExecutor to parallel process frames
#         max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(fetch_image, {"image": video_element, **process_info}, image_patch_size)
#                 for video_element in ele["video"]
#             ]
#             image_list = [future.result() for future in futures]

#         nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
#         if len(image_list) < nframes:
#             image_list.extend([image_list[-1]] * (nframes - len(image_list)))

#         sample_fps = ele.get("sample_fps", 2.0)
#         video = torch.stack([
#             torch.from_numpy(np.array(image).transpose(2, 0, 1))
#             for image in image_list
#         ])

#         # fake video metadata
#         raw_fps = process_info.pop("raw_fps", sample_fps)
#         logger.info(f"actual used sample_fps: {sample_fps}, raw_fps: {raw_fps}")
#         video_metadata = dict(
#             fps=raw_fps,
#             frames_indices=[i for i in range(len(video))],
#             total_num_frames=(nframes / sample_fps) * raw_fps,
#         )

#     nframes, _, height, width = video.shape
#     min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
#     total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
#     max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
#     max_pixels_supposed = ele.get("max_pixels", max_pixels)
#     if max_pixels_supposed > max_pixels:
#         logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
#     max_pixels = min(max_pixels_supposed, max_pixels)
#     if "resized_height" in ele and "resized_width" in ele:
#         resized_height, resized_width = smart_resize(
#             ele["resized_height"],
#             ele["resized_width"],
#             factor=image_factor,
#         )
#     else:
#         resized_height, resized_width = smart_resize(
#             height,
#             width,
#             factor=image_factor,
#             min_pixels=min_pixels,
#             max_pixels=max_pixels,
#         )
#     video = transforms.functional.resize(
#         video,
#         [resized_height, resized_width],
#         interpolation=InterpolationMode.BICUBIC,
#         antialias=True,
#     ).float()

#     final_video = (video, video_metadata) if return_video_metadata else video
#     if return_video_sample_fps:
#         return final_video, sample_fps
#     return final_video

# JJ: copy from source code for reference
# class ProcessorMixin(PushToHubMixin):

#     def apply_chat_template(
#         self,
#         conversation: list[dict[str, str]] | list[list[dict[str, str]]],
#         chat_template: str | None = None,
#         **kwargs: Unpack[AllKwargsForChatTemplate],
#     ) -> str:
#         """
#         Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
#         conversations to turn them into a single tokenizable string.

#         The input is expected to be in the following format, where each message content is a list consisting of text and
#         optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
#         `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
#                     {"type": "text", "text": "Please describe this image in detail."},
#                 ],
#             },
#         ]

#         Args:
#             conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
#                 The conversation to format.
#             chat_template (`Optional[str]`, *optional*):
#                 The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
#                 chat template is used.
#         """
#         if chat_template is None:
#             if isinstance(self.chat_template, dict) and "default" in self.chat_template:
#                 chat_template = self.chat_template["default"]
#             elif isinstance(self.chat_template, dict):
#                 raise ValueError(
#                     'The processor has multiple chat templates but none of them are named "default". You need to specify'
#                     " which one to use by passing the `chat_template` argument. Available templates are: "
#                     f"{', '.join(self.chat_template.keys())}"
#                 )
#             elif self.chat_template is not None:
#                 chat_template = self.chat_template
#             else:
#                 raise ValueError(
#                     "Cannot use apply_chat_template because this processor does not have a chat template."
#                 )
#         else:
#             if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
#                 # It's the name of a template, not a full template string
#                 chat_template = self.chat_template[chat_template]
#             else:
#                 # It's a template string, render it directly
#                 pass

#         # Check if tokenizer is fast - use backend attribute if available, otherwise fall back to class name
#         is_tokenizers_fast = False
#         if hasattr(self, "tokenizer"):
#             if hasattr(self.tokenizer, "backend"):
#                 is_tokenizers_fast = self.tokenizer.backend == "tokenizers"
#             else:
#                 # Fallback to class name check
#                 is_tokenizers_fast = self.tokenizer.__class__.__name__.endswith("Fast")

#         if kwargs.get("continue_final_message", False):
#             if kwargs.get("add_generation_prompt", False):
#                 raise ValueError(
#                     "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
#                 )
#             if kwargs.get("return_assistant_tokens_mask", False):
#                 raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

#         if kwargs.get("return_assistant_tokens_mask", False):
#             if not is_tokenizers_fast:
#                 raise ValueError(
#                     "`return_assistant_tokens_mask` is not possible with slow tokenizers. Make sure you have `tokenizers` installed. "
#                     "If the error persists, open an issue to support a Fast tokenizer for your model."
#                 )
#             else:
#                 kwargs["return_offsets_mapping"] = True  # force offset mapping so we can infer token boundaries

#         # Fill sets of kwargs that should be used by jinja template, filtering out kwargs used in `processor.__call__`
#         # NOTE: we don't only filter but also set the default values here. Without default values, we can remove it
#         template_kwargs = {}
#         for key in AllKwargsForChatTemplate.__annotations__["template_kwargs"].__annotations__:
#             kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__["template_kwargs"]
#             default_value = getattr(kwarg_type_defaults, key, None)
#             value = kwargs.pop(key, default_value)
#             if value is not None and not isinstance(value, dict):
#                 template_kwargs[key] = value

#         # Pass unprocessed custom kwargs
#         template_kwargs.update(kwargs)

#         # Set the sampling rate to load the audio files if user hasn't already passed with `kwargs`
#         if "sampling_rate" not in template_kwargs:
#             if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "sampling_rate"):
#                 template_kwargs["sampling_rate"] = self.feature_extractor.sampling_rate
#             else:
#                 template_kwargs["sampling_rate"] = 16_000

#         if isinstance(conversation, (list, tuple)) and (
#             isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
#         ):
#             is_batched = True
#             conversations = conversation
#         else:
#             is_batched = False
#             conversations = [conversation]

#         tokenize = template_kwargs.pop("tokenize", False)
#         return_dict = template_kwargs.pop("return_dict", True)

#         if tokenize:
#             batch_images, batch_videos = [], []
#             batch_audios = []
#             for conversation in conversations:
#                 images, videos = [], []
#                 for message in conversation:
#                     visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
#                     audio_fnames = [
#                         content[key]
#                         for content in message["content"]
#                         for key in ["audio", "url", "path"]
#                         if key in content and content["type"] == "audio"
#                     ]
#                     image_fnames = [
#                         vision_info[key]
#                         for vision_info in visuals
#                         for key in ["image", "url", "path", "base64"]
#                         if key in vision_info and vision_info["type"] == "image"
#                     ]
#                     images.extend(image_fnames)
#                     video_fnames = [
#                         vision_info[key]
#                         for vision_info in visuals
#                         for key in ["video", "url", "path"]
#                         if key in vision_info and vision_info["type"] == "video"
#                     ]
#                     videos.extend(video_fnames)

#                     # Audio models do not accept nested list of audios (yet!) so we construct a flat input audio list
#                     if not template_kwargs["load_audio_from_video"]:
#                         for fname in audio_fnames:
#                             batch_audios.append(load_audio(fname, sampling_rate=template_kwargs["sampling_rate"]))
#                     else:
#                         for fname in video_fnames:
#                             batch_audios.append(load_audio(fname, sampling_rate=template_kwargs["sampling_rate"]))

#                 # Currently all processors can accept nested list of batches, but not flat list of visuals
#                 # So we'll make a batched list of images and let the processor handle it
#                 batch_images.append(images)
#                 batch_videos.append(videos)

#         special_tokens_map = {}
#         if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "special_tokens_map"):
#             special_tokens = self.tokenizer.special_tokens_map
#             # Filter out tokens that conflict with template kwargs
#             special_tokens_map = {k: v for k, v in special_tokens.items() if k not in template_kwargs}

#         prompt, generation_indices = render_jinja_template(
#             conversations=conversations,
#             chat_template=chat_template,
#             **template_kwargs,  # different flags such as `return_assistant_mask`
#             **special_tokens_map,  # tokenizer special tokens are used by some templates
#         )

#         if not is_batched:
#             prompt = prompt[0]

#         if tokenize:
#             # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
#             # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
#             # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
#             # everything internally. The below line is to keep BC for that and be able to work with model that have
#             # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
#             # without actionable solution for users
#             single_prompt = prompt[0] if is_batched else prompt
#             if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
#                 kwargs["add_special_tokens"] = False

#             # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`fps`
#             # sampling should not done for BC.
#             if "do_sample_frames" not in kwargs and (
#                 kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
#             ):
#                 kwargs["do_sample_frames"] = True

#             images_exist = any((im is not None) for im_list in batch_images for im in im_list)
#             videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
#             out = self(
#                 text=prompt,
#                 images=batch_images if images_exist else None,
#                 videos=batch_videos if videos_exist else None,
#                 audio=batch_audios if batch_audios else None,
#                 **kwargs,
#             )

#             if return_dict:
#                 if template_kwargs.get("return_assistant_tokens_mask", False):
#                     assistant_masks = []
#                     offset_mapping = out.pop("offset_mapping")
#                     input_ids = out["input_ids"]
#                     for i in range(len(input_ids)):
#                         current_mask = [0] * len(input_ids[i])
#                         offsets = offset_mapping[i]
#                         offset_starts = [start for start, end in offsets]
#                         for assistant_start_char, assistant_end_char in generation_indices[i]:
#                             start_pos = bisect.bisect_left(offset_starts, assistant_start_char)
#                             end_pos = bisect.bisect_left(offset_starts, assistant_end_char)

#                             if not (
#                                 start_pos >= 0
#                                 and start_pos < len(offsets)
#                                 and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
#                             ):
#                                 # start_token is out of bounds maybe due to truncation.
#                                 continue
#                             # Ensure end_pos is also within bounds
#                             if end_pos > len(input_ids[i]):
#                                 end_pos = len(input_ids[i])
#                             for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
#                                 current_mask[token_id] = 1
#                         assistant_masks.append(current_mask)
#                     out["assistant_masks"] = assistant_masks
#                     out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
#                 return out
#             else:
#                 return out["input_ids"]
#         return prompt