from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import transformers

@dataclass
class ModelArguments:
    model_type: str = field(default="spatial-mllm")  # spatial-mllm, qwen2.5-vl, qwen2-vl
    vggt_checkpoints_path: Optional[str] = field(default="checkpoints/VGGT-1B/model.safetensors")
    spatial_embeds_layer_idx: int = field(default=-1)
    connector_type: str = field(default="mlp_add")  # mlp_add, mlp_cat, cross_attn

    pretrained_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_spatial_encoder: bool = field(default=False)
    tune_mm_connector: bool = field(default=False)
    
    # JJ: 4D Pose RoPE config
    use_pose_rope: bool = field(default=False, metadata={"help": "Enable 4D Pose-aware RoPE (P+T+H+W) instead of 3D mRoPE (T+H+W)"})
    pose_enc_type: str = field(default="PTHW", metadata={"help": "Pose encoding type ('PTHW', 'PHW', or 'THW')"})
    mrope_section: Optional[List[int]] = field(default=None, metadata={"help": "Custom mrope_section (e.g., [16, 24, 24] for 3D or [8, 8, 24, 24] for 4D)"})

    # JJ : LVSM integration config
    enforce_lvsm: bool = field(default=True, metadata={"help": "Enable LVSM NVS branch. Set False to fallback to base model."})
    lvsm_checkpoint_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained LVSM .pt checkpoint"})
    lvsm_config_path: Optional[str] = field(default=None, metadata={"help": "Path to LVSM YAML config (unused for now, defaults are hardcoded)"})
    nvs_loss_weight: float = field(default=1.0, metadata={"help": "Weight for NVS loss in combined loss"})
    num_target_views: int = field(default=4, metadata={"help": "Number of target views for NVS decoding"})
    lvsm_l2_weight: float = field(default=1.0, metadata={"help": "L2 loss weight in NVS loss"})
    lvsm_perceptual_weight: float = field(default=0.5, metadata={"help": "Perceptual loss weight in NVS loss"})
    lvsm_lpips_weight: float = field(default=0.0, metadata={"help": "LPIPS loss weight in NVS loss"})
    vgg_weight_file: str = field(default="./metric_checkpoint/imagenet-vgg-verydeep-19.mat", metadata={"help": "Path to VGG .mat weights for perceptual loss"})
    tune_mm_connector_lvsm: bool = field(default=True, metadata={"help": "Whether to train connector_lvsm (should always be True)"})

    # JJ :  NVS related fine-tuning config (mostly for ablation/debugging purposes; can be left as default for normal training)
    tune_lvsm_decoder: bool = field(default=True, metadata={"help": "Unfreeze LVSM transformer_blocks + image_token_decoder (keep image_tokenizer frozen)"})
    nvs_img_log_interval: int = field(default=50, metadata={"help": "Log rendered vs GT images to wandb every N steps (0=disable)"})
    nvs_target_pool: str = field(default="nvs", metadata={"help": "NVS target pool: 'nvs' (novel only), 'input' (input frames only), 'all' (input + novel)"})
    lvsm2qwen_type: str = field(default="linear", metadata={"help": "Phase-3 adapter type (LVSM→QwenVL). Currently only 'linear'."})
    llm2lvsm_type: str = field(default="linear", metadata={"help": "Phase-5 adapter type (LLM→LVSM). Currently only 'linear'."})
    vlm2context_adapt_strategy: str = field(
        default="patch_residual",
        metadata={"help": "Phase-5 context adaptation: 'film' (current) or 'patch_residual' (ctx=base+g*delta_patch)."},
    )

    # JJ : SDPA output gating (arxiv 2505.06708) — element-wise sigmoid gate after attention output
    enable_sdpa_gating: bool = field(default=False, metadata={"help": "Add learned sigmoid gate after SDPA output in every decoder layer. Enhances cross-view reasoning."})

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    # JJ: used for overfitting/debugging when set to a small number, or can be left as None for full dataset
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, truncate the shuffled training set to the first N samples."},
    )
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    # If set, treat the input video frames as if they were sampled at this FPS (nominal FPS).
    # Used to compute the temporal spacing (second_per_grid_ts) for RoPE, especially when videos
    # are already provided as pre-extracted frames and the original FPS is unknown/unreliable.
    video_frame_fps: Optional[int] = field(default=None)
    # JJ : NVS target frame loading (novel view synthesis)
    nvs_enabled: bool = field(default=False, metadata={"help": "Enable loading novel target frames for NVS loss"})
    # JJ : Enable loading precomputed pose(current is enforcenbr_after_step1_f16) package (.pt per video) instead of online pose/frame sampling.
    use_pre_compute_pose: bool = field(default=False, metadata={"help": "Use precomputed pose/index package from disk"})
    precompute_pose_source: str = field(default="vggt", metadata={"help": "Precomputed pose source. Supported: 'vggt'"})
    precompute_pose_root: str = field(default="", metadata={"help": "Root dir containing precomputed pose .pt files"})
    # JJ : Temporal-merge-aware real-neighbour sampling (not used if use_pre is true since precompute can already enforce this by design    )
    sampling_enforce_real_neighbour: bool = field(default=False, metadata={"help": "Sample N/2 anchors uniformly + N/2 real neighbours instead of N uniform frames"})
    neighbour_mode: str = field(default="random", metadata={"help": "Neighbour direction: 'before', 'after', or 'random'"})
    neighbour_max_step: int = field(default=1, metadata={"help": "Max frame offset for neighbour; actual step randomly sampled from [1, max_step]"})



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # JJ : Separate lr for LVSM adaptor (connector_lvsm only; lvsm_model uses base lr)
    lvsm_adaptor_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for connector_lvsm. lvsm_model (decoder) uses base --learning_rate instead."})
