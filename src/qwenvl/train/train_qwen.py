# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import os
import pathlib
import shutil
import sys
from pathlib import Path

# add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Trainer,
    TrainerCallback, # used for monitor intermidiate grad norms for LVSM adapter diagnosis
)

import src.qwenvl.train.trainer
from src.qwenvl.data.data_qwen import make_supervised_data_module
from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration
from src.qwenvl.preprocessor.image_processing_qwen2_vl import Qwen2VLImageProcessorModified
from src.qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from src.qwenvl.train.trainer import replace_qwen2_vl_attention_class


# jj: Log minimal per-bridge (llm2lvsm lvsm2llm) grad norms at pre-optimizer-step for LVSM adaptor diagnosis.
class LVSMBridgeGradMonitorCallback(TrainerCallback):
    def __init__(self, log_interval: int = 10):
        self.log_interval = max(1, int(log_interval))

    @staticmethod
    def _unwrap_model(model):
        cur = model
        while hasattr(cur, "module"):
            cur = cur.module
        return cur

    @staticmethod
    def _global_grad_norm(params):
        total = None
        for p in params:
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                sq = g.pow(2).sum()
                total = sq if total is None else total + sq
        if total is None:
            return 0.0
        return torch.sqrt(total).item()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or not args.should_log:
            return
        if (state.global_step + 1) % self.log_interval != 0:
            return
        if args.process_index != 0:
            return

        root_model = self._unwrap_model(model)
        connector = getattr(root_model, "connector_lvsm", None)
        if connector is None:
            return

        lvsm2llm_params = []
        llm2lvsm_params = []
        vggt2qwen_params = []# 16384-dim merged VGGT geo tokens can have large gradients during early training, so monitor separately for diagnosis.
        for name, p in connector.named_parameters():
            if name.startswith("lvsm_norm.") or name.startswith("lvsm_proj.") or name == "nvs_gate":
                lvsm2llm_params.append(p)
            elif (
                name.startswith("view_norm.")
                or name.startswith("view_proj.")
                or name.startswith("gamma_head.")
                or name.startswith("beta_head.")
                or name == "mod_gate"
                or name.startswith("cross_attn.")
            ):
                llm2lvsm_params.append(p)
        # JJ: Monitor optional VGGT->Qwen bridge module gradients together with existing bridge diagnostics.
        for module_name in ("vggt_geo_norm", "vggt_geo_proj"):
            module = getattr(root_model, module_name, None)
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad:
                        vggt2qwen_params.append(p)

        try:
            import wandb
            if wandb.run is None:
                return
            wandb.log(
                {
                    "grad/lvsm2llm_grad_norm": self._global_grad_norm(lvsm2llm_params),
                    "grad/llm2lvsm_grad_norm": self._global_grad_norm(llm2lvsm_params),
                    "grad/vggt2qwen_grad_norm": self._global_grad_norm(vggt2qwen_params),
                },
                commit=False,
            )
        except ImportError:
            return


# JJ: Add reproducibility control
def set_seed_for_reproducibility(seed=42):
    """
    Set seed for reproducibility across Python, NumPy, PyTorch, and CUDA.
    
    Args:
        seed: Random seed value (default: 42)
    """
    import random
    import numpy as np
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # CUDA deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch DataLoader worker seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    print(f"[INFO] ========================================")
    print(f"[INFO] Reproducibility seed set to: {seed}")
    print(f"[INFO] - torch.backends.cudnn.deterministic = True")
    print(f"[INFO] - torch.backends.cudnn.benchmark = False")
    print(f"[INFO] ========================================")
    
    return seed_worker  # Return for use in DataLoader


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_connector:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    if hasattr(model, "spatial_encoder"):
        if model_args.tune_mm_spatial_encoder:
            for n, p in model.spatial_encoder.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.spatial_encoder.named_parameters():
                p.requires_grad = False

    if hasattr(model, "connector"):
        if model_args.tune_mm_connector:
            for n, p in model.connector.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.connector.named_parameters():
                p.requires_grad = False

    # JJ : LVSM components
    if hasattr(model, "connector_lvsm"):
        if model_args.tune_mm_connector_lvsm:
            for n, p in model.connector_lvsm.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.connector_lvsm.named_parameters():
                p.requires_grad = False
    # JJ: Keep VGGT->Qwen bridge trainability aligned with LVSM adaptor switch for reproducible ablation behavior.
    for module_name in ("vggt_geo_norm", "vggt_geo_proj"):
        module = getattr(model, module_name, None)
        if module is not None:
            for n, p in module.named_parameters():
                p.requires_grad = bool(model_args.tune_mm_connector_lvsm)

    if hasattr(model, "lvsm_model"):
        # JJ : LVSM base: freeze everything first
        for n, p in model.lvsm_model.named_parameters():
            p.requires_grad = False
        # JJ : Resolve LVSM trainable-module whitelist with strict sanity check.
        # Keep legacy `tune_lvsm_decoder=False` behavior for backward compatibility when list is default.
        allowed_lvsm_modules = {
            "transformer_blocks": getattr(model.lvsm_model, "transformer_blocks", None),
            "transformer_input_layernorm": getattr(model.lvsm_model, "transformer_input_layernorm", None),
            "image_token_decoder": getattr(model.lvsm_model, "image_token_decoder", None),
            "image_tokenizer": getattr(model.lvsm_model, "image_tokenizer", None),
            "target_pose_tokenizer": getattr(model.lvsm_model, "target_pose_tokenizer", None),
        }
        default_lvsm_modules = ["transformer_blocks", "transformer_input_layernorm", "image_token_decoder"]
        requested_lvsm_modules = list(getattr(model_args, "lvsm_trainable_modules", default_lvsm_modules))
        requested_lvsm_modules = [m.strip() for m in requested_lvsm_modules if isinstance(m, str) and m.strip()]

        invalid_modules = sorted(set(requested_lvsm_modules) - set(allowed_lvsm_modules.keys()))
        if invalid_modules:
            raise ValueError(
                f"Invalid --lvsm_trainable_modules: {invalid_modules}. "
                f"Allowed: {sorted(allowed_lvsm_modules.keys())}"
            )

        # Deduplicate while preserving order for stable logging/reproducibility.
        unique_requested_modules = []
        for module_name in requested_lvsm_modules:
            if module_name not in unique_requested_modules:
                unique_requested_modules.append(module_name)

        if not getattr(model_args, "tune_lvsm_decoder", False) and unique_requested_modules == default_lvsm_modules:
            print("[INFO] tune_lvsm_decoder=False detected; override lvsm_trainable_modules to [] for backward compatibility.")
            unique_requested_modules = []

        for module_name in unique_requested_modules:
            module = allowed_lvsm_modules[module_name]
            if module is None:
                raise ValueError(f"LVSM module `{module_name}` not found in model.lvsm_model")
            for n, p in module.named_parameters():
                p.requires_grad = True
        model._lvsm_trainable_modules_resolved = unique_requested_modules
        print(f"[INFO] LVSM trainable modules: {unique_requested_modules}")

    if hasattr(model, "nvs_loss_fn"):
        # JJ : NVS loss modules are always frozen
        for n, p in model.nvs_loss_fn.named_parameters():
            p.requires_grad = False

    # JJ : SDPA gating params — always trainable when present
    _gate_count = 0
    for n, p in model.named_parameters():
        if "sdpa_gate" in n:
            p.requires_grad = True
            _gate_count += 1
    if _gate_count > 0:
        print(f"[INFO] SDPA gating: {_gate_count} gate params set to trainable")


def get_model(model_args, data_args, training_args, attn_implementation="flash_attention_2"):
    # JJ : Custom spatial MLLM with LVSM integration
    if model_args.model_type.lower() == "custom-spatial-mllm-lvsm":
        from src.custom_qwenvl.model.custom_spatial_mllm_lvsm import (
            CustomSpatialMLLMLVSMConfig,
            CustomSpatialMLLMLVSMForConditionalGeneration,
        )

        lvsm_config = {
            "enforce_LVSM": model_args.enforce_lvsm,
            "lvsm_checkpoint_path": model_args.lvsm_checkpoint_path,
            "nvs_loss_weight": model_args.nvs_loss_weight,
            "num_target_views": model_args.num_target_views,
            "l2_loss_weight": model_args.lvsm_l2_weight,
            "perceptual_loss_weight": model_args.lvsm_perceptual_weight,
            "lpips_loss_weight": model_args.lvsm_lpips_weight,
            "vgg_weight_file": model_args.vgg_weight_file,
            "nvs_img_log_interval": model_args.nvs_img_log_interval,
            "nvs_target_pool": getattr(model_args, 'nvs_target_pool', 'nvs'),
            "lvsm2qwen_type": getattr(model_args, 'lvsm2qwen_type', 'linear'),
            "llm2lvsm_type": getattr(model_args, 'llm2lvsm_type', 'linear'),
            "pluker_token_only_for_lvsm2llm": getattr(model_args, 'pluker_token_only_for_lvsm2llm', False),
            "vlm2context_adapt_strategy": getattr(model_args, 'vlm2context_adapt_strategy', 'patch_residual'),
        }

        spatial_mllm_lvsm_config = CustomSpatialMLLMLVSMConfig.from_pretrained(
            model_args.pretrained_model_name_or_path,
            spatial_config={
                "img_size": 518,
                "patch_size": 14,
                "embed_dim": 1024,
            },
            connector_config={
                "connector_type": model_args.connector_type,
                "spatial_embeds_layer_idx": model_args.spatial_embeds_layer_idx,
            },
            lvsm_config=lvsm_config,
        )
        # JJ : SDPA output gating flag — propagate to config so decoder layers can read it
        spatial_mllm_lvsm_config.enable_sdpa_gating = getattr(model_args, 'enable_sdpa_gating', False)
        model = CustomSpatialMLLMLVSMForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            config=spatial_mllm_lvsm_config,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        # JJ : Load VGGT weights (after from_pretrained, same as base model)
        model.spatial_encoder.load_pretrained_weights(model_args.vggt_checkpoints_path)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        model.spatial_encoder.to(device=device, dtype=dtype)
        # JJ : Load LVSM pretrained weights AFTER from_pretrained
        # Must be done here because post_init() in __init__ re-initializes
        # all nn.Linear modules, destroying weights loaded during __init__.
        model.load_lvsm_checkpoint()
        print(f"[INFO] LVSM checkpoint loaded from {model_args.lvsm_checkpoint_path}")
        model.lvsm_model.to(device=device, dtype=dtype)

        image_processor = Qwen2VLImageProcessorModified.from_pretrained(
            model_args.pretrained_model_name_or_path,
        )

    # JJ: Custom spatial MLLM with custom decoder
    elif model_args.model_type.lower() == "custom-spatial-mllm":
        from src.custom_qwenvl.model.custom_spatial_mllm import (
            CustomSpatialMLLMConfig,
            CustomSpatialMLLMForConditionalGeneration,
        )

        spatial_mllm_config = CustomSpatialMLLMConfig.from_pretrained(
            model_args.pretrained_model_name_or_path,
            spatial_config={
                "img_size": 518,
                "patch_size": 14,
                "embed_dim": 1024,
            },
            connector_config={
                "connector_type": model_args.connector_type,
                "spatial_embeds_layer_idx": model_args.spatial_embeds_layer_idx,
            },
        )
        # JJ : SDPA output gating flag
        spatial_mllm_config.enable_sdpa_gating = getattr(model_args, 'enable_sdpa_gating', False)
        model = CustomSpatialMLLMForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            config=spatial_mllm_config,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        # # load VGGT weights
        if "ct" not in model_args.model_type.lower():
            model.spatial_encoder.load_pretrained_weights(model_args.vggt_checkpoints_path)
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            model.spatial_encoder.to(device=device, dtype=dtype)

        image_processor = Qwen2VLImageProcessorModified.from_pretrained(
            model_args.pretrained_model_name_or_path,
        )
    elif "spatial-mllm" == model_args.model_type.lower():
        spatial_mllm_config = SpatialMLLMConfig.from_pretrained(
            model_args.pretrained_model_name_or_path,
            spatial_config={
                "img_size": 518,
                "patch_size": 14,
                "embed_dim": 1024,
            },
            connector_config={
                "connector_type": model_args.connector_type,
                "spatial_embeds_layer_idx": model_args.spatial_embeds_layer_idx,
            },
        )
        model = SpatialMLLMForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            config=spatial_mllm_config,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        if "ct" not in model_args.model_type.lower():
            model.spatial_encoder.load_pretrained_weights(model_args.vggt_checkpoints_path)
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            model.spatial_encoder.to(device=device, dtype=dtype)

        image_processor = Qwen2VLImageProcessorModified.from_pretrained(
            model_args.pretrained_model_name_or_path,
        )
    elif "qwen2.5-vl" == model_args.model_type.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        image_processor = AutoProcessor.from_pretrained(
            model_args.pretrained_model_name_or_path,
            use_fast=True,
        ).image_processor
    # else:
    elif "qwen2" == model_args.model_type.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.pretrained_model_name_or_path,
            use_fast=True,
        )
    else:
        raise NotImplementedError
    return model, image_processor


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # JJ: Set seed for reproducibility
    seed_worker = set_seed_for_reproducibility(seed=training_args.seed)

    model, image_processor = get_model(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        attn_implementation=attn_implementation,
    )
    data_args.image_processor = image_processor
    data_args.model_type = model_args.model_type

    # 🆕 NEW: Apply Pose RoPE monkey patch for custom-spatial-mllm
    if "custom-spatial-mllm" in model_args.model_type.lower() and model_args.use_pose_rope:
        from src.custom_qwenvl.model.custom_spatial_mllm_pose_rope import patch_model_with_pose_rope
        
        # Print user-level configuration before patching
        print(f"[Training] 🔧 Applying Pose RoPE configuration:")
        print(f"[Training]    - pose_enc_type: {model_args.pose_enc_type}")
        print(f"[Training]    - mrope_section: {model_args.mrope_section if model_args.mrope_section else 'default (will be determined by pose_enc_type)'}")
        
        model = patch_model_with_pose_rope(
            model,
            use_pose_rope=True,
            pose_enc_type=model_args.pose_enc_type,
            mrope_section=model_args.mrope_section,  # 🆕 NEW: Pass custom mrope_section if provided
            # Note: All Temporal & Pose parameters are inherited from model.__init__
        )
        
        # Dynamic message based on actual pose_enc_type
        if model_args.pose_enc_type == "PTHW":
            dims_desc = "4D Pose-aware RoPE (P+T+H+W)"
        elif model_args.pose_enc_type == "PHW":
            dims_desc = "3D Pose-aware RoPE (P+H+W, ignore temporal)"
        elif model_args.pose_enc_type == "THW":
            dims_desc = "3D standard mRoPE (T+H+W, ignore pose)"
        else:
            dims_desc = f"RoPE with pose_enc_type={model_args.pose_enc_type}"
        
        print(f"[Training] ✅ Monkey patch applied: Model now uses {dims_desc}")
        
        # 🆕 NEW: Save Pose RoPE config to model.config for checkpoint persistence
        if not hasattr(model.config, 'pose_rope_config'):
            model.config.pose_rope_config = {}
        model.config.pose_rope_config['use_pose_rope'] = True
        model.config.pose_rope_config['pose_enc_type'] = model_args.pose_enc_type
        model.config.pose_rope_config['mrope_section'] = model.config.rope_scaling["mrope_section"]
        print(f"[Training] 💾 Saved Pose RoPE config to model.config for checkpoint persistence")
        print(f"[Training]    - use_pose_rope: {model.config.pose_rope_config['use_pose_rope']}")
        print(f"[Training]    - pose_enc_type: {model.config.pose_rope_config['pose_enc_type']}")
        print(f"[Training]    - mrope_section: {model.config.pose_rope_config['mrope_section']}")
    elif "custom-spatial-mllm" in model_args.model_type.lower() and not model_args.use_pose_rope:
        print(f"[Training] ℹ️  Using standard 3D mRoPE (T+H+W)")
        # 🆕 NEW: Save config even when not using Pose RoPE
        if not hasattr(model.config, 'pose_rope_config'):
            model.config.pose_rope_config = {}
        model.config.pose_rope_config['use_pose_rope'] = False
        model.config.pose_rope_config['pose_enc_type'] = None
        model.config.pose_rope_config['mrope_section'] = model.config.rope_scaling.get("mrope_section", [16, 24, 24])

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # JJ : Print module-level trainable parameters status
    model.visual.print_trainable_parameters()
    model.model.print_trainable_parameters()
    if hasattr(model, "spatial_encoder"):
        model.spatial_encoder.print_trainable_parameters()
    if hasattr(model, "connector"):
        model.connector.print_trainable_parameters()
    # JJ : Print LVSM connector trainable params
    if hasattr(model, "connector_lvsm"):
        total_p = sum(p.numel() for p in model.connector_lvsm.parameters())
        train_p = sum(p.numel() for p in model.connector_lvsm.parameters() if p.requires_grad)
        print(f"[connector_lvsm] Total: {total_p:,}, Trainable: {train_p:,}")
    # JJ : Print LVSM model trainable params
    if hasattr(model, "lvsm_model"):
        total_p = sum(p.numel() for p in model.lvsm_model.parameters())
        train_p = sum(p.numel() for p in model.lvsm_model.parameters() if p.requires_grad)
        print(f"[lvsm_model] Total: {total_p:,}, Trainable: {train_p:,}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # JJ: For custom-spatial-mllm, wrap collator to remove position_ids
    # This forces the model to recompute position_ids with custom RoPE logic
    if "custom-spatial-mllm" in model_args.model_type.lower():
        original_collator = data_module['data_collator']
        
        def custom_spatial_mllm_collator_wrapper(instances):
            batch = original_collator(instances)
            batch.pop('position_ids', None)  # Remove if exists
            return batch
        
        data_module['data_collator'] = custom_spatial_mllm_collator_wrapper
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[LVSMBridgeGradMonitorCallback(log_interval=10)], # Add LVSM grad monitor callback
        **data_module,
    )

    # JJ : Force non-reentrant gradient checkpointing for DDP compatibility.
    # Reentrant checkpointing (default) re-runs forward during backward, which
    # triggers DDP all-reduce hooks twice per parameter → "marked ready twice" error.
    # Non-reentrant mode avoids this by using saved_tensors_hooks instead.
    # Setting here because shell cannot reliably pass JSON to HfArgumentParser.
    if training_args.gradient_checkpointing:
        trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # JJ : Print total parameters count after Trainer initialization
    # Note: Must be done after Trainer init because DeepSpeed ZeRO-3 wraps the model
    # and parameters are not fully accessible before that
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    trainer.train()

    trainer.save_state()

    # JJ : Handle HuggingFace model path for chat_template.json
    # Previous version (only works for local paths):
    # source_path = os.path.join(model_args.pretrained_model_name_or_path, "chat_template.json")
    # template_path = os.path.join(training_args.output_dir, "chat_template.json")
    # shutil.copy2(source_path, template_path)
    
    try:
        from huggingface_hub import hf_hub_download
        # Try to download from HuggingFace if it's a HF model ID
        source_path = hf_hub_download(
            repo_id=model_args.pretrained_model_name_or_path,
            filename="chat_template.json",
            repo_type="model"
        )
    except Exception as e:
        # Fallback to local path if not a HF model or file doesn't exist
        source_path = os.path.join(model_args.pretrained_model_name_or_path, "chat_template.json")
        if not os.path.exists(source_path):
            logging.warning(f"chat_template.json not found at {source_path}, skipping copy. Error: {e}")
            source_path = None
    
    if source_path and os.path.exists(source_path):
        template_path = os.path.join(training_args.output_dir, "chat_template.json")
        shutil.copy2(source_path, template_path)
        logging.info(f"Copied chat_template.json from {source_path} to {template_path}")
    else:
        logging.warning("chat_template.json not found, skipping copy")

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
