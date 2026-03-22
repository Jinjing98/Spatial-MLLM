import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLModel
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length, is_sagemaker_mp_enabled
from transformers.trainer_utils import seed_worker

import datasets


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def print_trainable_parameters_connector(self) -> None:
    """
    Prints the trainable status of all connector components
    """
    # Check trainable status of merger module
    is_connector_trainable = any(param.requires_grad for param in self.connector.parameters())

    # Print results
    print(f"Connector Module Trainable: {is_connector_trainable}")


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            visual_merger_parameters = [
                name for name, _ in opt_model.visual.merger.named_parameters(prefix="visual.merger")
            ]
            connector_parameters = [name for name, _ in opt_model.connector.named_parameters(prefix="connector")]
            projector_parameters = visual_merger_parameters + connector_parameters
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [name for name, _ in opt_model.visual.named_parameters()]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        # JJ : Separate lr group for LVSM adaptor (connector_lvsm only).
        # lvsm_model (decoder) stays in default group → uses base --learning_rate.
        lvsm_adaptor_lr = getattr(self.args, 'lvsm_adaptor_lr', None)
        if lvsm_adaptor_lr is not None and lvsm_adaptor_lr > 0:
            # JJ: Collect LVSM adaptor params (connector_lvsm + optional VGGT->Qwen bridge) into the same LR bucket.
            lvsm_param_ids = set()
            for module_name in ('connector_lvsm', 'vggt_geo_norm', 'vggt_geo_proj'):
                module = getattr(opt_model, module_name, None)
                if module is not None:
                    for p in module.parameters():
                        if p.requires_grad:
                            lvsm_param_ids.add(id(p))

            if lvsm_param_ids:
                # Remove LVSM params from existing groups
                for group in optimizer_grouped_parameters:
                    group["params"] = [p for p in group["params"] if id(p) not in lvsm_param_ids]

                # Collect LVSM params split by decay / no-decay
                lvsm_decay_params = []
                lvsm_no_decay_params = []
                for n, p in opt_model.named_parameters():
                    if id(p) in lvsm_param_ids:
                        if n in decay_parameters:
                            lvsm_decay_params.append(p)
                        else:
                            lvsm_no_decay_params.append(p)

                if lvsm_decay_params:
                    optimizer_grouped_parameters.append({
                        "params": lvsm_decay_params,
                        "weight_decay": self.args.weight_decay,
                        "lr": lvsm_adaptor_lr,
                    })
                if lvsm_no_decay_params:
                    optimizer_grouped_parameters.append({
                        "params": lvsm_no_decay_params,
                        "weight_decay": 0.0,
                        "lr": lvsm_adaptor_lr,
                    })
                print(f"[Optimizer] LVSM adaptor param groups: lr={lvsm_adaptor_lr}, "
                      f"decay={len(lvsm_decay_params)}, no_decay={len(lvsm_no_decay_params)}")

        # JJ : Optional separate lr group for LVSM trainable modules selected by model_args.lvsm_trainable_modules.
        # Default None keeps old behavior: these params stay in base --learning_rate group.
        lvsm_decoder_lr = getattr(self.args, 'lvsm_decoder_lr', None)
        if lvsm_decoder_lr is not None and lvsm_decoder_lr > 0:
            allowed_prefix_by_module = {
                "transformer_blocks": "lvsm_model.transformer_blocks.",
                "transformer_input_layernorm": "lvsm_model.transformer_input_layernorm.",
                "image_token_decoder": "lvsm_model.image_token_decoder.",
                "image_tokenizer": "lvsm_model.image_tokenizer.",
                "target_pose_tokenizer": "lvsm_model.target_pose_tokenizer.",
            }
            default_lvsm_modules = ["transformer_blocks", "transformer_input_layernorm", "image_token_decoder"]
            requested_lvsm_modules = list(getattr(opt_model, "_lvsm_trainable_modules_resolved", default_lvsm_modules))
            requested_lvsm_modules = [m.strip() for m in requested_lvsm_modules if isinstance(m, str) and m.strip()]
            invalid_modules = sorted(set(requested_lvsm_modules) - set(allowed_prefix_by_module.keys()))
            if invalid_modules:
                raise ValueError(
                    f"Invalid lvsm_trainable_modules in model config: {invalid_modules}. "
                    f"Allowed: {sorted(allowed_prefix_by_module.keys())}"
                )

            # Deduplicate while preserving order for stable optimizer diagnostics.
            unique_requested_modules = []
            for module_name in requested_lvsm_modules:
                if module_name not in unique_requested_modules:
                    unique_requested_modules.append(module_name)
            lvsm_decoder_prefixes = tuple(
                allowed_prefix_by_module[module_name] for module_name in unique_requested_modules
            )
            lvsm_decoder_param_ids = {
                id(p)
                for n, p in opt_model.named_parameters()
                if p.requires_grad and n.startswith(lvsm_decoder_prefixes)
            }

            if lvsm_decoder_param_ids:
                # Remove LVSM decoder params from existing groups first to avoid duplication.
                for group in optimizer_grouped_parameters:
                    group["params"] = [p for p in group["params"] if id(p) not in lvsm_decoder_param_ids]

                lvsm_decoder_decay_params = []
                lvsm_decoder_no_decay_params = []
                for n, p in opt_model.named_parameters():
                    if id(p) in lvsm_decoder_param_ids:
                        if n in decay_parameters:
                            lvsm_decoder_decay_params.append(p)
                        else:
                            lvsm_decoder_no_decay_params.append(p)

                if lvsm_decoder_decay_params:
                    optimizer_grouped_parameters.append({
                        "params": lvsm_decoder_decay_params,
                        "weight_decay": self.args.weight_decay,
                        "lr": lvsm_decoder_lr,
                    })
                if lvsm_decoder_no_decay_params:
                    optimizer_grouped_parameters.append({
                        "params": lvsm_decoder_no_decay_params,
                        "weight_decay": 0.0,
                        "lr": lvsm_decoder_lr,
                    })
                print(f"[Optimizer] LVSM decoder param groups: lr={lvsm_decoder_lr}, modules={unique_requested_modules}, "
                      f"decay={len(lvsm_decoder_decay_params)}, no_decay={len(lvsm_decoder_no_decay_params)}")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# Apply monkey patches
Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
