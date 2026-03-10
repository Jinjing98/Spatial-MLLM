"""
JJ : NVS Loss for LVSM integration.

Adapted from LVSM's loss.py but:
  - Removed torch.distributed dependency (for single-GPU training)
  - Provides user-adjustable weight entries for L2, Perceptual, LPIPS
  - Raises error if VGG weight file not found (user must provide)
"""

import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from easydict import EasyDict as edict

logger = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    """
    JJ : VGG-based perceptual loss (no distributed dependency).
    Adapted from LVSM's PerceptualLoss.
    """
    
    def __init__(self, weight_file="./metric_checkpoint/imagenet-vgg-verydeep-19.mat"):
        super().__init__()
        # JJ : Store path so we can reload after model.to(bf16)
        self._weight_file = weight_file
        self.vgg = self._build_vgg()
        self._load_weights(weight_file)
        self._setup_feature_blocks()
    
    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        return model.eval()
    
    def _load_weights(self, weight_file):
        """Load pre-trained VGG weights from .mat file."""
        import scipy.io
        
        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                f"[NVSLoss] VGG weight file not found: {weight_file}\n"
                f"Please download from: https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat\n"
                f"And place it at: {os.path.abspath(weight_file)}"
            )
        
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # JJ : .contiguous() after .permute() to avoid non-contiguous tensors in save_pretrained
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1).contiguous()
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i]).contiguous()
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)
    
    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i + 1]]))
            self.blocks.append(block.eval())
        for param in self.blocks.parameters():
            param.requires_grad = False
        # JJ : Delete self.vgg to avoid shared tensor issue in save_pretrained
        # (self.blocks already holds all needed layers; keeping self.vgg would
        #  create duplicate parameter references that safetensors rejects)
        del self.vgg

    def reload_vgg_float32(self):
        """
        JJ : Reload VGG weights from the original .mat file in float32.

        WHY: from_pretrained(torch_dtype=bf16) calls model.to(bf16), which
        converts VGG weights from float64 → bf16.  Even though we cast back to
        float32 in forward(), the roundtrip float64→bf16→float32 loses precision
        and causes NaN in deep VGG blocks.

        This method rebuilds the VGG, reloads original .mat weights (float64),
        slices into the existing blocks, and explicitly converts to float32.
        Must be called AFTER from_pretrained / model.to(bf16).
        """
        # Rebuild full VGG and load original .mat weights (float64)
        vgg_tmp = self._build_vgg()
        self._load_weights_into(vgg_tmp, self._weight_file)

        # Re-slice into blocks (same indices as _setup_feature_blocks)
        output_indices = [0, 4, 9, 14, 23, 32]
        for i in range(len(output_indices) - 1):
            new_block = nn.Sequential(
                *list(vgg_tmp.features[output_indices[i]:output_indices[i + 1]])
            )
            self.blocks[i] = new_block.eval().float()  # float64 → float32
        for param in self.blocks.parameters():
            param.requires_grad = False
        del vgg_tmp
        logger.warning("[INFO] PerceptualLoss VGG weights reloaded from .mat in float32")

    @staticmethod
    def _load_weights_into(vgg_model, weight_file):
        """JJ : Load VGG .mat weights into a given VGG model (no self mutation)."""
        import scipy.io
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                weights = torch.from_numpy(
                    vgg_layers[layer_idx][0][0][2][0][0]
                ).permute(3, 2, 0, 1).contiguous()
                vgg_model.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                biases = torch.from_numpy(
                    vgg_layers[layer_idx][0][0][2][0][1]
                ).view(filter_sizes[i]).contiguous()
                vgg_model.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)
    
    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean
    
    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))
    
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, pred_img, target_img):
        """Compute perceptual loss between prediction and target."""
        # JJ : Disable autocast and force float32.
        # After reload_vgg_float32(), blocks are already float32,
        # so block.float() below is a no-op safety net.
        pred_img = pred_img.float()
        target_img = target_img.float()

        # JJ [DIAG] : Trace NaN source inside VGG blocks
        # logger.warning(f"[DIAG-Percep] inputs: pred finite={torch.isfinite(pred_img).all().item()} "
        #       f"range=[{pred_img.min():.3f},{pred_img.max():.3f}] | "
        #       f"target finite={torch.isfinite(target_img).all().item()} "
        #       f"range=[{target_img.min():.3f},{target_img.max():.3f}]")

        target_p = self._preprocess_images(target_img)
        pred_p = self._preprocess_images(pred_img)
        
        target_features = [target_p]
        pred_features = [pred_p]
        for i, block in enumerate(self.blocks):
            block_f32 = block.float()  # no-op if already float32 (safety net)
            tf = block_f32(target_features[-1])
            pf = block_f32(pred_features[-1])
            # # JJ [DIAG] : Report NaN at block level
            # if not torch.isfinite(tf).all() or not torch.isfinite(pf).all():
            #     logger.warning(f"[DIAG-Percep] Block {i} → NaN/Inf detected! "
            #           f"target finite={torch.isfinite(tf).all().item()} "
            #           f"pred finite={torch.isfinite(pf).all().item()} "
            #           f"input_range=[{target_features[-1].min():.3f},{target_features[-1].max():.3f}]")
            # else:
            #     logger.warning(f"[DIAG-Percep] Block {i} OK: "
            #           f"range=[{tf.min():.3f},{tf.max():.3f}]")
            target_features.append(tf)
            pred_features.append(pf)
        
        e0 = self._compute_error(target_features[0], pred_features[0])
        e1 = self._compute_error(target_features[1], pred_features[1]) / 2.6
        e2 = self._compute_error(target_features[2], pred_features[2]) / 4.8
        e3 = self._compute_error(target_features[3], pred_features[3]) / 3.7
        e4 = self._compute_error(target_features[4], pred_features[4]) / 5.6
        e5 = self._compute_error(target_features[5], pred_features[5]) * 10 / 1.5
        
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
        return total_loss


class NVSLoss(nn.Module):
    """
    JJ : NVS Loss combining L2, Perceptual, and optional LPIPS losses.
    
    All loss modules are frozen (no learnable params).
    User can adjust weights via config.
    """
    
    def __init__(
        self,
        l2_weight=1.0,
        perceptual_weight=0.5,
        lpips_weight=0.0,
        vgg_weight_file="./metric_checkpoint/imagenet-vgg-verydeep-19.mat",
    ):
        super().__init__()
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_weight = lpips_weight
        
        if self.perceptual_weight > 0.0:
            self.perceptual_loss_module = PerceptualLoss(weight_file=vgg_weight_file)
            self.perceptual_loss_module.eval()
            for param in self.perceptual_loss_module.parameters():
                param.requires_grad = False
        
        if self.lpips_weight > 0.0:
            import lpips
            self.lpips_loss_module = lpips.LPIPS(net="vgg")
            self.lpips_loss_module.eval()
            for param in self.lpips_loss_module.parameters():
                param.requires_grad = False
    
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, rendering, target):
        """
        JJ : Compute NVS loss.
        
        Autocast disabled — all NVS loss computed in float32 to avoid bf16 NaN.
        
        Args:
            rendering: [B, V, 3, H, W] - rendered images, value range [0, 1]
            target: [B, V, 3, H, W] - ground truth target images, value range [0, 1]
        
        Returns:
            loss_metrics: edict with loss, l2_loss, perceptual_loss, lpips_loss, psnr
        """
        # JJ : Cast to float32 — bf16 NVS loss causes NaN
        rendering = rendering.float()
        target = target.float()
        
        b, v, _, h, w = rendering.size()
        rendering_flat = rendering.reshape(b * v, 3, h, w)
        target_flat = target.reshape(b * v, 3, h, w)
        
        # L2 loss
        l2_loss = torch.tensor(1e-8, device=rendering.device)
        if self.l2_weight > 0.0:
            l2_loss = F.mse_loss(rendering_flat, target_flat)
        
        psnr = -10.0 * torch.log10(l2_loss)
        
        # Perceptual loss
        perceptual_loss = torch.tensor(0.0, device=rendering.device)
        if self.perceptual_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering_flat, target_flat)
        
        # LPIPS loss
        lpips_loss = torch.tensor(0.0, device=rendering.device)
        if self.lpips_weight > 0.0:
            lpips_loss = self.lpips_loss_module(
                rendering_flat * 2.0 - 1.0, target_flat * 2.0 - 1.0
            ).mean()
        
        # Combined loss
        loss = (
            self.l2_weight * l2_loss
            + self.perceptual_weight * perceptual_loss
            + self.lpips_weight * lpips_loss
        )
        
        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            perceptual_loss=perceptual_loss,
            lpips_loss=lpips_loss,
        )
        return loss_metrics
