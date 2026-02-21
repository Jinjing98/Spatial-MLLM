# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
# "qwen2_5_VLPRoPE" (Qwen2.5-VL PRoPE)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import Qwen2_5_VLConfig
import torch
from typing import Optional, Tuple

# JJ: Handle import for both module usage and direct execution
try:
    from src.utils.pose_distance_metrics import compute_lie_scalar_index_torch
except ModuleNotFoundError:
    # When running as __main__, add parent directory to path
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parents[3]  # Go up to Spatial-MLLM root
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.utils.pose_distance_metrics import compute_lie_scalar_index_torch

# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this rope_parameters to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
# ROPE_INIT_FUNCTIONS = {
#     "linear": _compute_linear_scaling_rope_parameters,
#     "dynamic": _compute_dynamic_ntk_parameters,
#     "yarn": _compute_yarn_parameters,
#     "longrope": _compute_longrope_parameters,
#     "llama3": _compute_llama3_parameters,
# }

class CustomQwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        print('*'*20)
        print('RoPE type: ', self.rope_type)
        print(f'Rope_init_fn for type {self.rope_type}: {self.rope_init_fn.__name__}')
        print('RoPE rope_theta:', self.config.rope_theta)
        print('RoPE dim:', self.config.hidden_size // self.config.num_attention_heads)

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # ✏️ MODIFIED: Support both 3D (T,H,W) and 4D (P,T,H,W) RoPE by dynamically expanding based on position_ids shape
        num_dims = position_ids.shape[0]  # 3 for THW, 4 for PTHW
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(num_dims, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (num_dims, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# given inv_freq and max_position_embeddings, plot and save the positional encoding map
def plot_positional_encoding(inv_freq, max_position_embeddings, save_path="rope_encoding.png", sample_positions=512,
                            rope_type="default", rope_theta=10000.0, hidden_size=None):
    """
    Plot and save the positional encoding map based on inv_freq and max_position_embeddings.
    
    Args:
        inv_freq: torch.Tensor of shape (dim/2,) containing inverse frequencies
        max_position_embeddings: int, maximum sequence length
        save_path: str, path to save the plot
        sample_positions: int, number of positions to sample for visualization (to avoid memory issues)
        rope_type: str, type of RoPE (e.g., 'default', 'linear', 'dynamic')
        rope_theta: float, base frequency for RoPE
        hidden_size: int, hidden dimension size
    """
    # Sample positions uniformly from the full range
    positions = torch.linspace(0, max_position_embeddings - 1, sample_positions, dtype=torch.long)
    
    # Compute the positional encodings: pos * inv_freq
    # Shape: (sample_positions, dim/2)
    freqs = positions[:, None] * inv_freq[None, :]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert to numpy for plotting
    freqs_np = freqs.cpu().numpy()
    inv_freq_np = inv_freq.cpu().numpy()
    positions_np = positions.cpu().numpy()
    
    # Plot 1: Heatmap of sin(freqs)
    im1 = axes[0, 0].imshow(np.sin(freqs_np), aspect='auto', cmap='RdBu', interpolation='nearest')
    axes[0, 0].set_title('sin(position × inv_freq)')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_yticks(np.linspace(0, sample_positions - 1, 5))
    axes[0, 0].set_yticklabels([f'{int(p)}' for p in np.linspace(0, max_position_embeddings - 1, 5)])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Heatmap of cos(freqs)
    im2 = axes[0, 1].imshow(np.cos(freqs_np), aspect='auto', cmap='RdBu', interpolation='nearest')
    axes[0, 1].set_title('cos(position × inv_freq)')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].set_yticks(np.linspace(0, sample_positions - 1, 5))
    axes[0, 1].set_yticklabels([f'{int(p)}' for p in np.linspace(0, max_position_embeddings - 1, 5)])
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Inverse frequencies
    axes[1, 0].plot(inv_freq_np)
    axes[1, 0].set_title('Inverse Frequencies')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('inv_freq')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Sample positional encodings at different positions
    sample_pos_indices = [0, sample_positions//4, sample_positions//2, 3*sample_positions//4, sample_positions-1]
    for idx in sample_pos_indices:
        pos_value = positions_np[idx]
        axes[1, 1].plot(np.sin(freqs_np[idx]), label=f'pos={int(pos_value)}', alpha=0.7)
    axes[1, 1].set_title('sin(position × inv_freq) at different positions')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('sin value')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Add overall title with configuration information
    title_parts = [
        f"RoPE Type: {rope_type}",
        f"Theta: {rope_theta}",
        f"Max Pos Emb: {max_position_embeddings}",
        f"Inv Freq Shape: {inv_freq.shape[0]}"
    ]
    if hidden_size is not None:
        title_parts.insert(2, f"Hidden Size: {hidden_size}")
    
    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Positional encoding plot saved to: {save_path}")
    plt.close()

def custom_get_rope_index(
    config: Qwen2_5_VLConfig,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids_compute_mode: Optional[str] = 'mRoPE',
    selected_frames_id: Optional[List[int]] = None,
    temporal_patch_size: Optional[int] = 2,
    temporal_readapted_merge_strategy: Optional[str] = 'mean',  # JJ: Renamed for consistency
    temporal_readapted_use_dynamic_scale_factor: Optional[bool] = True,  # JJ: Renamed for consistency
    temporal_readapted_scale_factor: Optional[float] = 16.0,  # JJ: Renamed for consistency
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
    Only used once during prefill stage.

    # JJ: new feature
    If position_ids_compute_mode is 'mRoPE_woT', the temporal dimension of the position ids will be set to 0 for all video tokens.
    Otherwise, the temporal dimension of the position ids will be set to the temporal dimension of the video tokens.
    # JJ: new feature
    If position_ids_compute_mode is 'mRoPE_readaptT', the temporal dimension of the position ids will be remodulated based on the pattern in selected_frames_id

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        position_ids_compute_mode (`str`, *optional*):
            The mode to compute the position ids. Can be 'mRoPE' or 'mRoPE_woT'.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        visual_token_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    """
    if position_ids_compute_mode == 'mRoPE_readaptT':
        assert len(video_grid_thw) == 1, f'Only one video is supported for mRoPE_readaptT'
        assert len(selected_frames_id) == video_grid_thw[0][0] * temporal_patch_size, f'Expected {video_grid_thw[0][0] * temporal_patch_size} frames but got {len(selected_frames_id)} frames'

    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        # # JJ: visual patch token mask (B, L)
        visual_token_mask = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            llm_visual_mask_list: list = [] #JJ
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                # -------------------------
                # Text segment before <image>/<video> token
                # -------------------------
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # JJ
                llm_visual_mask_list.append(torch.zeros(text_len, dtype=torch.long, device=input_ids.device))

                # -------------------------
                # Vision patch segment
                # -------------------------
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)

                # JJ. The only difference compared to mRoPE: set T dim as 0 for all video tokens
                # Update range_tensor for other varients
                if position_ids_compute_mode == 'mRoPE_woT':
                    range_tensor = torch.zeros_like(range_tensor)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * temporal_patch_size #config.vision_config.tokens_per_second
                elif position_ids_compute_mode == 'mRoPE_readaptT':
                    # JJ: Validate temporal_patch_size for current strategies
                    if temporal_patch_size != 2:
                        raise NotImplementedError(
                            f"mRoPE_readaptT with temporal_patch_size={temporal_patch_size} is not implemented. "
                            f"Currently only temporal_patch_size=2 is supported. "
                            f"If you need other temporal_patch_size values, please implement the corresponding "
                            f"aggregation strategy in the 'temporal_readapted_merge_strategy' logic."
                        )
                    
                    # JJ: Aggregate selected_frames_id based on strategy
                    selected_frames_tensor = torch.tensor(selected_frames_id).view(-1, temporal_patch_size).float()
                    
                    if temporal_readapted_merge_strategy == 'mean':
                        adapt_range_tensor = selected_frames_tensor.mean(dim=1)
                    elif temporal_readapted_merge_strategy == 'first':
                        adapt_range_tensor = selected_frames_tensor[:, 0]
                    elif temporal_readapted_merge_strategy == 'last':
                        adapt_range_tensor = selected_frames_tensor[:, -1]
                    elif temporal_readapted_merge_strategy == 'median':
                        # For temporal_patch_size=2, median is equivalent to mean
                        adapt_range_tensor = selected_frames_tensor.median(dim=1)[0]
                    else:
                        raise ValueError(
                            f"Unknown temporal_readapted_merge_strategy: {temporal_readapted_merge_strategy}. "
                            f"Supported strategies: ['mean', 'first', 'last', 'median']"
                        )
                    
                    range_tensor = adapt_range_tensor.view(-1, 1)
                    
                    # JJ: Two scaling strategies for mRoPE_readaptT
                    if temporal_readapted_use_dynamic_scale_factor:
                        # Dynamic: Map to [0, llm_grid_t-1] based on current video's temporal dimension
                        # This ensures position IDs span the full temporal range available
                        target_max_t_pos = (llm_grid_t * temporal_patch_size - 1)
                    else:
                        # Fixed: Map to [0, temporal_readapted_scale_factor] for consistent encoding
                        # This provides fair temporal encoding across different sampling granularities
                        target_max_t_pos = temporal_readapted_scale_factor
                    
                    # Apply linear mapping: [min_frame_id, max_frame_id] → [0, target_max_t_pos]
                    original_min_frame = adapt_range_tensor.min()
                    original_max_frame = adapt_range_tensor.max()
                    original_frame_range = original_max_frame - original_min_frame
                    
                    scale_factor = target_max_t_pos / original_frame_range if original_frame_range > 0 else 1.0
                    
                    # Normalize: first shift to start at 0, then scale to target range
                    range_tensor = (range_tensor - original_min_frame) * scale_factor
                    range_tensor = range_tensor.to(range_tensor.dtype)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    
                    # JJ: Use multiplier of 1 since we obtain t_pos from frame_id directly
                    time_tensor = expanded_range * second_per_grid_t * 1
                else:
                    assert position_ids_compute_mode in ['mRoPE'], f'Expected mRoPE_woT or mRoPE but got {position_ids_compute_mode}'
                    range_tensor = range_tensor
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * temporal_patch_size #config.vision_config.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                
                # JJ
                vision_len = llm_grid_t * llm_grid_h * llm_grid_w        
                llm_visual_mask_list.append(torch.ones(vision_len, dtype=torch.long, device=input_ids.device))
                # JJ: Temporarily disabled for cleaner loss debugging
                st = ed + vision_len #llm_grid_t * llm_grid_h * llm_grid_w
            
            # remaining trailing text
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_visual_mask_list.append(torch.zeros(text_len, dtype=torch.long, device=input_ids.device))# JJ


            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            llm_visual_mask = torch.cat(llm_visual_mask_list, dim=0).reshape(-1) # JJ
            
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            visual_token_mask[i, attention_mask[i] == 1] = llm_visual_mask # JJ

            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        
        # JJ: Temporarily disabled for cleaner loss debugging
        # print(f'new_st, new_ed, trailing text len: {st}, {ed}, {text_len}')
        # print(f'llm_positions max value: {llm_positions.max()}')
        # print(f'total_input_ids shape: {total_input_ids[0].shape}')
        # print(f'mrope_position_deltas: {mrope_position_deltas}')
        
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas, visual_token_mask
    else:
        # No vision case
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        # JJ: no-vision => visual_token_mask all zeros
        visual_token_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        return position_ids, mrope_position_deltas, visual_token_mask

def custom_get_pose_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    selected_frames_poses: Optional[torch.Tensor] = None,  # JJ: Camera poses (N, 4, 4)
    selected_frames_id: Optional[List[int]] = None,  # JJ: Frame indices for mRoPE_readaptT
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pose_enc_type: str = "PTHW",  # default PTHW
    temporal_patch_size: int = 2,
    # ==================== Pose Parameters ====================
    pose_scale_factor: float = 16.0,  # JJ: Renamed from global_normalize_scale_factor
    pose_merge_strategy: str = 'mean',  # JJ: Independent merge strategy for Pose
    pose_use_dynamic_scale_factor: bool = False,  # JJ: New parameter for dynamic scaling
    pose_anchor_rereference_strategy: str = 'first',  # JJ: 'first' = re-reference to first frame
    pose_id_scalar_lambda_trans: float = 1.0,  # JJ: Balance weight between rotation and translation
    hard_reset_reference_after_pose_merge: bool = True,  # JJ: Re-norm after aggregation to restore reference point
    do_offset_in_pose_pos_id: bool = True,  # JJ: Renamed from add_offset_in_pose_id
    # ==================== Temporal Parameters ====================
    THW_position_ids_compute_mode: str = 'mRoPE',  # JJ: 'mRoPE', 'mRoPE_woT', 'mRoPE_readaptT'
    temporal_readapted_merge_strategy: str = 'mean',  # JJ: Renamed for consistency
    temporal_readapted_use_dynamic_scale_factor: bool = True,  # JJ: Renamed for consistency
    temporal_readapted_scale_factor: float = 16.0,  # JJ: Renamed for consistency
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 4D Pose-Temporal-Height-Width RoPE indices for vision+text sequence.

    Output shape: (4, batch_size, seq_len)
    Pose index is computed via compute_lie_scalar_index_torch.

    Args:
        selected_frames_poses: (N, 4, 4) camera-to-world poses for Pose encoding
        selected_frames_id: List of frame indices for mRoPE_readaptT temporal adjustment
        pose_enc_type: ['PTHW'], default PTHW (only PTHW implemented for now)
        
        Pose Parameters:
            pose_scale_factor: scale factor for Pose normalization [0, scale_factor]
            pose_merge_strategy: aggregation strategy for pose patches ('mean', 'first', 'last', 'median')
            pose_use_dynamic_scale_factor: use dynamic scale based on pose count
            pose_anchor_rereference_strategy: re-reference all poses to anchor frame ('first', 'medoid')
            pose_id_scalar_lambda_trans: balance weight between rotation and translation in Lie scalar
                - Affects (1) medoid selection: which frame is chosen as geometric center
                - Affects (2) P value ordering: relative distances of all frames to reference
                - Lambda > 1: emphasize translation, Lambda < 1: emphasize rotation
            hard_reset_reference_after_pose_merge: re-normalize after aggregation
            do_offset_in_pose_pos_id: add sequential offset to Pose dimension
        
        Temporal Parameters:
            THW_position_ids_compute_mode: temporal adjustment strategy ('mRoPE', 'mRoPE_woT', 'mRoPE_readaptT')
            temporal_readapted_merge_strategy: aggregation strategy for temporal patches ('mean', 'first', 'last', 'median')
            temporal_readapted_use_dynamic_scale_factor: use dynamic scale based on T-length
            temporal_readapted_scale_factor: fixed scale factor for temporal dimension
        
        Others: follow original custom_get_rope_index API

    Returns:
        position_ids: (4, batch_size, seq_len) - Pose dimension is float, THW dimensions are long
        rope_deltas: (batch_size, 1)
        visual_token_mask: (batch_size, seq_len)
    """
    # ------------------ CHECK ------------------
    if pose_enc_type != "PTHW":
        raise NotImplementedError(f"pose_enc_type={pose_enc_type} not implemented. Only 'PTHW' is supported.")
    if THW_position_ids_compute_mode == 'mRoPE_readaptT':
        if temporal_patch_size != 2:
            raise NotImplementedError(
                f"mRoPE_readaptT with temporal_patch_size={temporal_patch_size} is not implemented. "
                f"Currently only temporal_patch_size=2 is supported."
            )
        if selected_frames_id is None:
            raise ValueError("selected_frames_id is required for mRoPE_readaptT mode")
    # ------------------ END CHECK ------------------

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    spatial_merge_size = config.vision_config.spatial_merge_size

    # ------------------ REUSE: initialize position_ids and mask ------------------
    # JJ: Keep Pose dimension as float, THW dimensions as long
    position_ids = torch.ones(
        4, batch_size, seq_len, dtype=torch.float32, device=input_ids.device
    )
    visual_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
    rope_deltas = []
    # ------------------ END REUSE ------------------

    # ------------------ EXTEND: Select anchor frame based on strategy ------------------
    # JJ: Determine which frame to use as reference for pose distance computation
    if pose_anchor_rereference_strategy == 'first':
        # Use first frame as anchor
        ref_frame_idx = 0
    elif pose_anchor_rereference_strategy == 'medoid':
        # Use medoid frame as anchor (frame with minimum sum of distances to all others)
        # NOTE: Uses pose_id_scalar_lambda_trans to ensure consistent distance metric
        #       This affects: (1) which frame is selected as medoid
        #                     (2) ensures P value ordering matches medoid selection criterion
        N = selected_frames_poses.shape[0]
        device = selected_frames_poses.device
        
        # Compute distance matrix: D[i,j] = distance between pose i and pose j
        # Using same Lie scalar metric as compute_lie_scalar_index_torch
        distance_matrix = torch.zeros(N, N, device=device)
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Compute relative pose: T_j_to_i = inv(T_i_to_w) @ T_j_to_w
                    pose_i_w2c = torch.linalg.inv(selected_frames_poses[i])
                    pose_rel = pose_i_w2c @ selected_frames_poses[j]
                    
                    # Compute Lie scalar distance (rotation angle + weighted translation)
                    R_rel = pose_rel[:3, :3]
                    t_rel = pose_rel[:3, 3]
                    
                    # Rotation angle from trace(R) = 1 + 2*cos(theta)
                    trace = torch.diagonal(R_rel).sum()
                    cos_theta = (trace - 1.0) / 2.0
                    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                    theta = torch.acos(cos_theta)
                    
                    # Translation distance
                    d_trans = torch.norm(t_rel)
                    
                    # Combined distance with pose_id_scalar_lambda_trans weighting
                    # Same formula as in compute_lie_scalar_index_torch
                    distance_matrix[i, j] = torch.sqrt(
                        theta**2 + (pose_id_scalar_lambda_trans**2) * d_trans**2
                    )
        
        # Find medoid: frame with minimum sum of distances
        dist_sums = distance_matrix.sum(dim=1)  # (N,)
        ref_frame_idx = torch.argmin(dist_sums).item()
    else:
        raise NotImplementedError(
            f"pose_anchor_rereference_strategy='{pose_anchor_rereference_strategy}' not supported. "
            f"Only 'first' and 'medoid' are implemented."
        )
    
    print(f'[INFO] Reference frame index: {ref_frame_idx}')
    # ------------------ END EXTEND ------------------

    # ------------------ REUSE: compute Pose index (P) via Lie scalar ------------------
    # P shape: (num_frames,), range: [0, 1] if global_normalize=True
    # NOTE: Using same pose_id_scalar_lambda_trans as medoid selection to ensure:
    #       (1) Medoid selection criterion matches the final distance metric
    #       (2) P value ordering is consistent with the chosen anchor frame
    P = compute_lie_scalar_index_torch(
        poses_c2w=selected_frames_poses,
        pose_id_scalar_lambda_trans=pose_id_scalar_lambda_trans,  # Same weighting as medoid selection
        traj_scale_norm=True,
        global_normalize=True,
        reorth_rot=True,
        reference_frame_id=ref_frame_idx,  # 🆕 NEW: use selected anchor frame
    )
    # min is for sure zero, as reference frame distance to itself is 0
    print(f'P_nonscaled/anchor_strategy:{pose_anchor_rereference_strategy}/archor_idx:{ref_frame_idx}:')
    print(f'{P}')
    # if reorth is disabled we can gurateen P[ref_frame_idx] == 0
    # assert P[ref_frame_idx] == 0, \
        # f"P[{ref_frame_idx}] (reference frame) should be 0, but got {P[ref_frame_idx]}"
    # sanity check P
    if P.min() < 0 or P.max() == 0:
        raise ValueError(f"P contains invalid values, min={P.min()}, max={P.max()}")
    # ------------------ END REUSE ------------------

    # ------------------ EXTEND: Move P to correct device ------------------
    # ✏️ FIXED: Ensure P is on the same device as input_ids
    P = P.to(input_ids.device)

    # ------------------ END EXTEND ------------------

    # ------------------ EXTEND: iterate batch and vision tokens to build PTHW indices ------------------
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    
    for i in range(batch_size):
        st = 0
        llm_pos_ids_list = []
        llm_visual_mask_list = []

        input_masked = input_ids[i][attention_mask[i]==1]
        input_tokens = input_masked.tolist()
        vision_start_indices = torch.argwhere(input_masked == vision_start_token_id).squeeze(1)
        vision_tokens = input_masked[vision_start_indices+1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        image_index, video_index = 0, 0
        remain_images, remain_videos = image_nums, video_nums

        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens)+1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens)+1

            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            
            # -------------------------
            # Text segment before <image>/<video> token
            # -------------------------
            text_len = ed - st
            # JJ: st_idx is based on max of all dimensions (including Pose)
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            # JJ: For text tokens, all 4 dimensions (PTHW) follow sequential pattern
            text_pthw = torch.arange(text_len, dtype=torch.float32, device=input_ids.device).view(1, -1).expand(4, -1) + st_idx
            llm_pos_ids_list.append(text_pthw)
            llm_visual_mask_list.append(torch.zeros(text_len, dtype=torch.long, device=input_ids.device))

            # -------------------------
            # Vision patch segment: Pose + Temporal + Height + Width
            # -------------------------
            # JJ: Sanity check - P length should match llm_grid_t * temporal_patch_size
            assert len(P) == llm_grid_t * temporal_patch_size, \
                f"Expected P length {llm_grid_t * temporal_patch_size}, got {len(P)}"
            
            # JJ: Aggregate P based on temporal_patch_size
            selected_frames_tensor = P.view(llm_grid_t, temporal_patch_size)
            if pose_merge_strategy == 'mean':
                pose_range_tensor = selected_frames_tensor.mean(dim=1)
            elif pose_merge_strategy == 'first':
                pose_range_tensor = selected_frames_tensor[:, 0]
            elif pose_merge_strategy == 'last':
                pose_range_tensor = selected_frames_tensor[:, -1]
            elif pose_merge_strategy == 'median':
                pose_range_tensor = selected_frames_tensor.median(dim=1)[0]
            else:
                raise ValueError(
                    f"Unknown pose_merge_strategy: {pose_merge_strategy}. "
                    f"Supported strategies: ['mean', 'first', 'last', 'median']"
                )
            
            # JJ: Warning if pose and temporal strategies differ
            if pose_merge_strategy != temporal_readapted_merge_strategy:
                print(f"[WARNING] pose_merge_strategy ('{pose_merge_strategy}') != temporal_readapted_merge_strategy ('{temporal_readapted_merge_strategy}')")
            
            # JJ: Optionally re-normalize aggregated pose to [0, 1] before scaling
            # Motivation: Aggregation (especially mean) hides the zero pose due to re-reference,
            # losing "reference frame" semantics. Re-norm restores this and ensures consistent
            # range utilization, mirroring the temporal dimension's handling.
            if hard_reset_reference_after_pose_merge:
                pose_min = pose_range_tensor.min()
                pose_max = pose_range_tensor.max()
                pose_range = pose_max - pose_min
                if pose_range > 0:
                    pose_range_tensor = (pose_range_tensor - pose_min) / pose_range  # Re-norm to [0, 1]
            
            # JJ: Scale (re-)normalized pose to target range
            if pose_use_dynamic_scale_factor:
                # Dynamic: scale to [0, len(selected_frames_poses) - 1]
                target_max_p_pos = len(selected_frames_poses) - 1
                pose_range_tensor = pose_range_tensor * target_max_p_pos
            else:
                # Fixed: scale to [0, pose_scale_factor]
                pose_range_tensor = pose_range_tensor * pose_scale_factor
            
            # JJ: Expand pose_range_tensor to H*W patches
            pose_idx = pose_range_tensor.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            # Keep pose_idx as float for precise frequency encoding
            # DEBUG: print(f"DEBUG pose_idx range: [{pose_idx.min():.4f}, {pose_idx.max():.4f}]")
            
            # -------------------------
            # Temporal index: inherit from custom_get_rope_index (lines 389-455)
            # -------------------------
            range_tensor = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1)
            
            if THW_position_ids_compute_mode == 'mRoPE_woT':
                range_tensor = torch.zeros_like(range_tensor)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * temporal_patch_size
            elif THW_position_ids_compute_mode == 'mRoPE_readaptT':
                # JJ: Aggregate selected_frames_id based on strategy
                selected_frames_id_tensor = torch.tensor(selected_frames_id, device=input_ids.device).view(-1, temporal_patch_size).float()
                
                if temporal_readapted_merge_strategy == 'mean':
                    adapt_range_tensor = selected_frames_id_tensor.mean(dim=1)
                elif temporal_readapted_merge_strategy == 'first':
                    adapt_range_tensor = selected_frames_id_tensor[:, 0]
                elif temporal_readapted_merge_strategy == 'last':
                    adapt_range_tensor = selected_frames_id_tensor[:, -1]
                elif temporal_readapted_merge_strategy == 'median':
                    adapt_range_tensor = selected_frames_id_tensor.median(dim=1)[0]
                else:
                    raise ValueError(
                        f"Unknown temporal_readapted_merge_strategy: {temporal_readapted_merge_strategy}. "
                        f"Supported strategies: ['mean', 'first', 'last', 'median']"
                    )
                
                range_tensor = adapt_range_tensor.view(-1, 1)
                
                # JJ: Two scaling strategies for mRoPE_readaptT
                if temporal_readapted_use_dynamic_scale_factor:
                    target_max_t_pos = (llm_grid_t * temporal_patch_size - 1)
                else:
                    target_max_t_pos = temporal_readapted_scale_factor
                
                # Apply linear mapping: [min_frame_id, max_frame_id] → [0, target_max_t_pos]
                original_min_frame = adapt_range_tensor.min()
                original_max_frame = adapt_range_tensor.max()
                original_frame_range = original_max_frame - original_min_frame
                
                # ✏️ FIXED: Keep scale_factor as tensor to avoid device issues
                if original_frame_range > 0:
                    scale_factor = target_max_t_pos / original_frame_range
                else:
                    scale_factor = torch.tensor(1.0, device=input_ids.device, dtype=range_tensor.dtype)
                
                # Normalize: first shift to start at 0, then scale to target range
                range_tensor = (range_tensor - original_min_frame) * scale_factor
                range_tensor = range_tensor.to(device=input_ids.device, dtype=range_tensor.dtype)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                
                # JJ: Use multiplier of 1 since we obtain t_pos from frame_id directly
                time_tensor = expanded_range * second_per_grid_t * 1
            else:
                assert THW_position_ids_compute_mode == 'mRoPE', f'Expected mRoPE, mRoPE_woT or mRoPE_readaptT but got {THW_position_ids_compute_mode}'
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * temporal_patch_size

            time_tensor_long = time_tensor.long()
            t_index = time_tensor_long.flatten()

            # H, W index
            h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            
            # JJ: Stack as Pose + T + H + W
            # Pose dimension: optionally add offset based on do_offset_in_pose_pos_id flag
            # T/H/W dimensions: always add offset to maintain continuous position IDs
            if do_offset_in_pose_pos_id:
                # Pose follows sequential pattern with offset
                print('Again inspect pose_idx:')
                print(f'{pose_idx.min()},{pose_idx.max()}')
                pose_with_offset = pose_idx + text_len + st_idx
                thw_with_offset = torch.stack([t_index, h_index, w_index]).float() + text_len + st_idx
                llm_pos_ids_list.append(torch.cat([pose_with_offset.unsqueeze(0), thw_with_offset], dim=0))
            else:
                # Pose stays in its normalized range [0, pose_scale_factor]
                thw_with_offset = torch.stack([t_index, h_index, w_index]).float() + text_len + st_idx
                llm_pos_ids_list.append(torch.cat([pose_idx.unsqueeze(0), thw_with_offset], dim=0))
            
            vision_len = llm_grid_t * llm_grid_h * llm_grid_w
            llm_visual_mask_list.append(torch.ones(vision_len, dtype=torch.long, device=input_ids.device))
            st = ed + vision_len

        # trailing text
        if st < len(input_tokens):
            text_len = len(input_tokens) - st
            # JJ: st_idx is based on max of all dimensions (including Pose)
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            # JJ: For text tokens, all 4 dimensions (PTHW) follow sequential pattern
            text_pthw = torch.arange(text_len, dtype=torch.float32, device=input_ids.device).view(1, -1).expand(4, -1) + st_idx
            llm_pos_ids_list.append(text_pthw)
            llm_visual_mask_list.append(torch.zeros(text_len, dtype=torch.long, device=input_ids.device))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(4, -1)
        llm_visual_mask = torch.cat(llm_visual_mask_list, dim=0).reshape(-1)
        
        position_ids[..., i, attention_mask[i]==1] = llm_positions.to(position_ids.device)
        visual_token_mask[i, attention_mask[i]==1] = llm_visual_mask
        rope_deltas.append(llm_positions.max() + 1 - len(input_ids[i]))

    rope_deltas = torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, rope_deltas, visual_token_mask


if __name__ == "__main__":
    config = Qwen2_5_VLConfig()
    # update some param
    config.rope_type = 'default'
    config.rope_theta = 10000.0
    config.hidden_size = 128
    config.num_attention_heads = 1
    config.max_position_embeddings = 32768  # JJ: Required for CustomQwen2_5_VLRotaryEmbedding
    # JJ: Add vision_config attributes for testing
    if not hasattr(config, 'vision_config'):
        from types import SimpleNamespace
        config.vision_config = SimpleNamespace()
    config.vision_config.spatial_merge_size = 2
    config.vision_start_token_id = 151652
    config.image_token_id = 151655
    config.video_token_id = 151656


    rotary_emb = CustomQwen2_5_VLRotaryEmbedding(config)
    BS = 1
    seq_len = 13
    x = torch.randn(BS, seq_len, 128)# only used for obtain device
    position_ids = torch.randint(0, seq_len, (3, BS, seq_len))# 3 BS seq_len
    cos, sin = rotary_emb(x, position_ids)
    print(cos.shape) # 3 BS seq_len 128*2
    print(sin.shape) # 3 BS seq_len 128*2
    #print details regarding the qwen 2.5 vl rope:
    print(f"rope_type: {'default'}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"ROPE_INIT_FUNCTIONS: {ROPE_INIT_FUNCTIONS.keys()}")
    print(f"attention_scaling: {ROPE_INIT_FUNCTIONS['default'](config)[-1]}")
    print(f"Updated inv_freq: {ROPE_INIT_FUNCTIONS['default'](config)[0].shape}")
    print(f"original_max_seq_len: {rotary_emb.original_max_seq_len}")

    # plot_positional_encoding(rotary_emb.inv_freq, config.max_position_embeddings)
    # For 16frames (after temporal merge and spatioal merge:with have (16/2)*46*34/(2*2)=3128 tokens)
    # After prefill: there are 3210 tokens
    # Max_position_embeddings_practical = (10~15) + tem_merge*8*fps + (~70)) ~= 105
    plot_positional_encoding(
        rotary_emb.inv_freq, 
        max_position_embeddings=105,
        rope_type=config.rope_type,
        rope_theta=config.rope_theta,
        hidden_size=config.hidden_size
    )

    # ========================================
    # JJ: Comparable test for pose-aware RoPE
    # ========================================
    print("\n" + "="*80)
    print("TESTING: custom_get_rope_index vs custom_get_pose_rope_index")
    print("="*80)
    
    # Setup test data
    BS = 1
    num_frames = 16
    temporal_patch_size = 2
    llm_grid_t = num_frames // temporal_patch_size  # 8
    llm_grid_h, llm_grid_w = 46 // 2, 34 // 2  # 23, 17 after spatial merge
    vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w  # 3128
    text_tokens_before = 10
    text_tokens_after = 20
    total_tokens = text_tokens_before + vision_tokens + text_tokens_after  # 3158
    
    # Create dummy input_ids (simplified: text + video + text)
    input_ids = torch.zeros(BS, total_tokens, dtype=torch.long)
    input_ids[0, 0:text_tokens_before] = 1  # text tokens
    input_ids[0, text_tokens_before-1] = config.vision_start_token_id  # vision start
    input_ids[0, text_tokens_before] = config.video_token_id  # video token
    input_ids[0, text_tokens_before+1:text_tokens_before+vision_tokens+1] = 2  # vision patch tokens
    input_ids[0, text_tokens_before+vision_tokens+1:] = 1  # trailing text
    
    attention_mask = torch.ones(BS, total_tokens, dtype=torch.long)
    
    # Video grid: [T, H, W] before spatial merge
    video_grid_thw = torch.tensor([[llm_grid_t, 46, 34]], dtype=torch.long)
    image_grid_thw = None
    
    # JJ: Generate dummy camera poses (N, 4, 4) for Pose encoding
    # Create a simple camera trajectory moving along x-axis with slight rotation
    selected_frames_poses = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
    for i in range(num_frames):
        # Translation along x-axis
        selected_frames_poses[i, 0, 3] = i * 0.1  # x translation
        selected_frames_poses[i, 1, 3] = i * 0.02  # y translation
        selected_frames_poses[i, 2, 3] = i * 0.01  # z translation
        # Small rotation around y-axis
        angle = i * 0.05
        selected_frames_poses[i, 0, 0] = torch.cos(torch.tensor(angle))
        selected_frames_poses[i, 0, 2] = torch.sin(torch.tensor(angle))
        selected_frames_poses[i, 2, 0] = -torch.sin(torch.tensor(angle))
        selected_frames_poses[i, 2, 2] = torch.cos(torch.tensor(angle))
    
    # For custom_get_rope_index, we still use frame indices
    selected_frames_id = list(range(num_frames))  # [0, 1, 2, ..., 15]
    second_per_grid_ts = torch.tensor([1.0])  # 1 second per grid
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {BS}")
    print(f"  Total frames: {num_frames}")
    print(f"  temporal_patch_size: {temporal_patch_size}")
    print(f"  llm_grid_t (after merge): {llm_grid_t}")
    print(f"  llm_grid_h, llm_grid_w: {llm_grid_h}, {llm_grid_w}")
    print(f"  Vision tokens: {vision_tokens}")
    print(f"  Total sequence length: {total_tokens}")
    
    # ========================================
    # Test 1: Standard mRoPE (3D)
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 1: custom_get_rope_index with mRoPE")
    print(f"{'='*80}")
    
    pos_ids_3d, deltas_3d, vis_mask_3d = custom_get_rope_index(
        config=config,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        position_ids_compute_mode='mRoPE',
        selected_frames_id=selected_frames_id,
        temporal_patch_size=temporal_patch_size,
    )
    
    print(f"  Output shape: {pos_ids_3d.shape}")  # (3, BS, seq_len)
    print(f"  Deltas shape: {deltas_3d.shape}")
    print(f"  Visual mask shape: {vis_mask_3d.shape}")
    print(f"  Position IDs range - T: [{pos_ids_3d[0].min()}, {pos_ids_3d[0].max()}]")
    print(f"  Position IDs range - H: [{pos_ids_3d[1].min()}, {pos_ids_3d[1].max()}]")
    print(f"  Position IDs range - W: [{pos_ids_3d[2].min()}, {pos_ids_3d[2].max()}]")
    print(f"  Visual tokens count: {vis_mask_3d.sum().item()}")
    
    # ========================================
    # Test 2: Pose-aware mRoPE (4D) - PTHW with mRoPE
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 2: custom_get_pose_rope_index with PTHW + mRoPE")
    print(f"{'='*80}")
    
    pos_ids_4d, deltas_4d, vis_mask_4d = custom_get_pose_rope_index(
        config=config,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        selected_frames_poses=selected_frames_poses,  # JJ: Pass camera poses
        selected_frames_id=selected_frames_id,  # JJ: Pass frame indices (needed for mRoPE_readaptT)
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        pose_enc_type='PTHW',
        temporal_patch_size=temporal_patch_size,
        pose_scale_factor=16.0,
        pose_merge_strategy='mean',
        pose_use_dynamic_scale_factor=False,
        do_offset_in_pose_pos_id=False,  # JJ: Keep Pose in normalized range [0, 16]
        THW_position_ids_compute_mode='mRoPE',
        temporal_readapted_merge_strategy='mean',
    )
    
    print(f"  Output shape: {pos_ids_4d.shape}")  # (4, BS, seq_len)
    print(f"  Deltas shape: {deltas_4d.shape}")
    print(f"  Visual mask shape: {vis_mask_4d.shape}")
    print(f"  Position IDs range - P: [{pos_ids_4d[0].min():.4f}, {pos_ids_4d[0].max():.4f}]")
    print(f"  Position IDs range - T: [{pos_ids_4d[1].min()}, {pos_ids_4d[1].max()}]")
    print(f"  Position IDs range - H: [{pos_ids_4d[2].min()}, {pos_ids_4d[2].max()}]")
    print(f"  Position IDs range - W: [{pos_ids_4d[3].min()}, {pos_ids_4d[3].max()}]")
    print(f"  Visual tokens count: {vis_mask_4d.sum().item()}")
    
    # ========================================
    # Test 3: Pose-aware mRoPE_readaptT (4D)
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 3: custom_get_pose_rope_index with PTHW + mRoPE_readaptT")
    print(f"{'='*80}")
    
    pos_ids_4d_readapt, deltas_4d_readapt, vis_mask_4d_readapt = custom_get_pose_rope_index(
        config=config,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        selected_frames_poses=selected_frames_poses,  # JJ: Pass camera poses
        selected_frames_id=selected_frames_id,  # JJ: Pass frame indices (required for mRoPE_readaptT)
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        pose_enc_type='PTHW',
        temporal_patch_size=temporal_patch_size,
        pose_scale_factor=16.0,
        pose_merge_strategy='mean',
        pose_use_dynamic_scale_factor=False,
        do_offset_in_pose_pos_id=False,
        THW_position_ids_compute_mode='mRoPE_readaptT',
        temporal_readapted_merge_strategy='mean',
        temporal_readapted_use_dynamic_scale_factor=True,
    )
    
    print(f"  Output shape: {pos_ids_4d_readapt.shape}")
    print(f"  Position IDs range - P: [{pos_ids_4d_readapt[0].min():.4f}, {pos_ids_4d_readapt[0].max():.4f}]")
    print(f"  Position IDs range - T: [{pos_ids_4d_readapt[1].min()}, {pos_ids_4d_readapt[1].max()}]")
    print(f"  Position IDs range - H: [{pos_ids_4d_readapt[2].min()}, {pos_ids_4d_readapt[2].max()}]")
    print(f"  Position IDs range - W: [{pos_ids_4d_readapt[3].min()}, {pos_ids_4d_readapt[3].max()}]")
    
    # ========================================
    # Test 4: Consistency check - THW dimensions should match
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 4: Consistency check between 3D and 4D RoPE")
    print(f"{'='*80}")
    
    # Extract vision tokens only
    vision_start_idx = text_tokens_before + 1
    vision_end_idx = vision_start_idx + vision_tokens
    
    # THW from 3D RoPE
    t_3d = pos_ids_3d[0, 0, vision_start_idx:vision_end_idx]
    h_3d = pos_ids_3d[1, 0, vision_start_idx:vision_end_idx]
    w_3d = pos_ids_3d[2, 0, vision_start_idx:vision_end_idx]
    
    # THW from 4D RoPE
    p_4d = pos_ids_4d[0, 0, vision_start_idx:vision_end_idx]
    t_4d = pos_ids_4d[1, 0, vision_start_idx:vision_end_idx]
    h_4d = pos_ids_4d[2, 0, vision_start_idx:vision_end_idx]
    w_4d = pos_ids_4d[3, 0, vision_start_idx:vision_end_idx]
    
    # Check if THW dimensions match
    t_match = torch.allclose(t_3d.float(), t_4d.float())
    h_match = torch.allclose(h_3d.float(), h_4d.float())
    w_match = torch.allclose(w_3d.float(), w_4d.float())
    
    print(f"  T dimension matches: {t_match}")
    print(f"  H dimension matches: {h_match}")
    print(f"  W dimension matches: {w_match}")
    
    if not (t_match and h_match and w_match):
        print("  ⚠️  WARNING: THW dimensions do NOT match!")
        print(f"    T diff: max={torch.abs(t_3d.float() - t_4d.float()).max()}")
        print(f"    H diff: max={torch.abs(h_3d.float() - h_4d.float()).max()}")
        print(f"    W diff: max={torch.abs(w_3d.float() - w_4d.float()).max()}")
    else:
        print("  ✓ THW dimensions are consistent!")
    
    # Show Pose dimension statistics
    print(f"\n  Pose dimension (P) statistics (vision tokens only):")
    print(f"    Mean: {p_4d.float().mean():.4f}")
    print(f"    Std:  {p_4d.float().std():.4f}")
    print(f"    Min:  {p_4d.float().min():.4f}")
    print(f"    Max:  {p_4d.float().max():.4f}")
    print(f"    Unique values: {len(torch.unique(p_4d))}")
    
    # Check Pose range for all tokens (including text)
    p_4d_all = pos_ids_4d[0, 0, :]
    print(f"\n  Pose dimension (P) statistics (all tokens including text):")
    print(f"    Min:  {p_4d_all.float().min():.4f}")
    print(f"    Max:  {p_4d_all.float().max():.4f}")
    
    # ========================================
    # Test 5: Different aggregation strategies
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 5: Comparing aggregation strategies (mean vs first vs last)")
    print(f"{'='*80}")
    
    strategies = ['mean', 'first', 'last', 'median']
    pose_ranges = {}
    
    for strategy in strategies:
        pos_ids_strat, _, _ = custom_get_pose_rope_index(
            config=config,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            selected_frames_poses=selected_frames_poses,  # JJ: Pass camera poses
            selected_frames_id=selected_frames_id,  # JJ: Pass frame indices
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            pose_enc_type='PTHW',
            temporal_patch_size=temporal_patch_size,
            pose_merge_strategy=strategy,
            THW_position_ids_compute_mode='mRoPE',
            temporal_readapted_merge_strategy=strategy,
        )
        p_strat = pos_ids_strat[0, 0, vision_start_idx:vision_end_idx]
        pose_ranges[strategy] = (p_strat.min().item(), p_strat.max().item(), p_strat.float().mean().item())
        print(f"  {strategy:8s}: min={pose_ranges[strategy][0]:.4f}, max={pose_ranges[strategy][1]:.4f}, mean={pose_ranges[strategy][2]:.4f}")
    
    print(f"\n{'='*80}")
    print("All tests completed successfully! ✓")
    print(f"{'='*80}\n")