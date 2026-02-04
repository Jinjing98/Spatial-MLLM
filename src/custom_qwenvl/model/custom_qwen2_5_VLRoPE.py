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
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
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
                elif position_ids_compute_mode == 'mRoPE_readaptT':
                    adapt_range_tensor = torch.tensor(selected_frames_id).view(-1, temporal_patch_size).float().mean(dim=1)
                    range_tensor = adapt_range_tensor.view(-1, 1)#.to(range_tensor.dtype)
                    # JJ FIXME: we do this for strict compariable
                    # scale factor to be comparibale with the sa img input [0,16] range
                    scale_factor = 16.0 / (adapt_range_tensor.max()-adapt_range_tensor.min()+1)
                    range_tensor = (range_tensor * scale_factor).to(range_tensor.dtype)
                else:
                    assert position_ids_compute_mode in ['mRoPE'], f'Expected mRoPE_woT or mRoPE but got {position_ids_compute_mode}'
                    range_tensor = range_tensor

                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                # JJ FIXME: set as 1 / temporal_patch_size should be conceptually correct!
                if position_ids_compute_mode == 'mRoPE_readaptT':
                    time_tensor = expanded_range * second_per_grid_t * 1 # *1 as here we abtain t_pos from frame_id directly.
                else:
                    assert position_ids_compute_mode in ['mRoPE_woT', 'mRoPE'], f'Expected mRoPE_woT or mRoPE but got {position_ids_compute_mode}'
                    time_tensor = expanded_range * second_per_grid_t * temporal_patch_size #config.vision_config.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()
                print('time_tensor_long',  time_tensor_long, t_index)

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                
                # JJ
                vision_len = llm_grid_t * llm_grid_h * llm_grid_w        
                llm_visual_mask_list.append(torch.ones(vision_len, dtype=torch.long, device=input_ids.device))
                print(f'*'*20)
                print(f'Meta at rope position_ids_compute_mode: {position_ids_compute_mode}')
                st = ed + vision_len #llm_grid_t * llm_grid_h * llm_grid_w
                print(f'ed, visual_len, text_len: {ed}, {vision_len}, {text_len}')
            
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
        
        print(f'new_st, new_ed, trailing text len: {st}, {ed}, {text_len}')
        print(f'llm_positions max value: {llm_positions.max()}')
        print(f'total_input_ids shape: {total_input_ids[0].shape}')
        print(f'mrope_position_deltas: {mrope_position_deltas}')
        
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

if __name__ == "__main__":
    config = Qwen2_5_VLConfig()
    # update some param
    config.rope_type = 'default'
    config.rope_theta = 10000.0
    config.rope_theta = 10000.0
    config.hidden_size = 128
    config.num_attention_heads = 1


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