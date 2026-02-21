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
)

import src.qwenvl.train.trainer
from src.qwenvl.data.data_qwen import make_supervised_data_module
from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration
from src.qwenvl.preprocessor.image_processing_qwen2_vl import Qwen2VLImageProcessorModified
from src.qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from src.qwenvl.train.trainer import replace_qwen2_vl_attention_class


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


def get_model(model_args, data_args, training_args, attn_implementation="flash_attention_2"):
    # JJ: Custom spatial MLLM with custom decoder
    if model_args.model_type.lower() == "custom-spatial-mllm":
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
    elif "spatial-mllm" in model_args.model_type.lower():
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
    elif "qwen2.5" in model_args.model_type.lower():
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
    else:
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
        model = patch_model_with_pose_rope(
            model,
            use_pose_rope=True,
            pose_enc_type=model_args.pose_enc_type,
            # Note: All Temporal & Pose parameters are inherited from model.__init__
        )
        print(f"[Training] ✅ Monkey patch applied: Model now uses 4D Pose-aware RoPE (P+T+H+W)")
    elif "custom-spatial-mllm" in model_args.model_type.lower() and not model_args.use_pose_rope:
        print(f"[Training] ℹ️  Using standard 3D mRoPE (T+H+W)")

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
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    
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
