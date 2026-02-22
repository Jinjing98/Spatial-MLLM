"""
DA3 Spatial Encoder - Wrapper for Depth Anything 3 model.

This module provides a DA3-based spatial encoder that extracts camera poses
(extrinsics + intrinsics) for pose-aware sampling and Pose RoPE.

Key Features:
- Compatible interface with VGGTSpatialEncoderPreTrainedModel
- Only extracts camera poses (no depth/intermediate features needed)
- Supports both cam_head and ray_head pose estimation
- Maintains temporal ordering for video sequences

Author: JJ
Date: 2026-02-21
"""

import sys
from pathlib import Path

# JJ: Add DA3 to path
DA3_PATH = Path(__file__).resolve().parents[3] / "submodule" / "Depth-Anything-3" / "src"
if str(DA3_PATH) not in sys.path:
    sys.path.insert(0, str(DA3_PATH))

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from typing import List, Optional, Tuple

try:
    # JJ: Try to import DA3 API wrapper (handles model loading from HF Hub)
    # Note: The API has some export dependencies (moviepy etc) but we don't need them
    import sys
    import warnings
    
    # Suppress moviepy-related import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from depth_anything_3.api import DepthAnything3
            DA3_API_AVAILABLE = True
        except ImportError:
            DA3_API_AVAILABLE = False
    
    # Always import core modules (these don't have moviepy dependency)
    from depth_anything_3.model.da3 import DepthAnything3Net
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.cfg import load_config, create_object
    from depth_anything_3.registry import MODEL_REGISTRY
    DA3_AVAILABLE = True
except ImportError as e:
    DA3_AVAILABLE = False
    DA3_API_AVAILABLE = False
    DA3_IMPORT_ERROR = str(e)
    print(f"[WARNING] DA3 core modules import failed: {e}")


class DA3SpatialEncoderConfig(PretrainedConfig):
    """
    Configuration for DA3 Spatial Encoder.
    
    Args:
        model_name: DA3 model variant
            Options: "da3-small", "da3-base", "da3-large", "da3-giant", "da3-large", "da3-giant"
        use_ray_pose: Use ray-based pose estimation (more accurate but slower)
            Default: False (use camera decoder)
        ref_view_strategy: Reference view selection strategy
            Options: "first", "middle", "saddle_balanced", "saddle_sim_range"
            Default: "first" (preserve temporal order for video sequences)
        process_res: Target resolution for image preprocessing
            Default: 504 (DA3 official recommendation, 504 = 14 × 36)
            Note: VGGT uses 518 (518 = 14 × 37). Both are multiples of patch_size=14.
        process_res_method: Resize method
            Options: "upper_bound_resize", "lower_bound_resize"
            Default: "upper_bound_resize" (resize so longer side = process_res)
            
    Note:
        For video sequences with temporal ordering, ALWAYS use "first" to maintain
        frame order consistency with selected_frames indices.
    """
    
    model_type = "da3_spatial_encoder"
    
    def __init__(
        self,
        model_name: str = "da3-large",
        use_ray_pose: bool = False,
        ref_view_strategy: str = "first",  # JJ: "first" to preserve temporal order
        process_res: int = 504,  # JJ: DA3 default (VGGT uses 518)
        process_res_method: str = "upper_bound_resize",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy
        assert self.ref_view_strategy in ["first"],f'TODO we should conduct extra pose conversion if we used the recommended saddle, since our code base expect c2w input not c2bestview input? but maybe we can directly proceed if we enable medoid?'
        self.process_res = process_res
        self.process_res_method = process_res_method


class DA3SpatialEncoderPreTrainedModel(PreTrainedModel):
    """
    DA3 Spatial Encoder for camera pose estimation.
    
    This encoder wraps the Depth Anything 3 model to extract camera poses from
    video frames. It provides a compatible interface with VGGTSpatialEncoderPreTrainedModel
    while leveraging DA3's superior pose estimation capabilities.
    
    Key Differences from VGGT:
    - DA3 directly outputs extrinsics [B, N, 3, 4] and intrinsics [B, N, 3, 3]
    - No need for pose_encoding_to_extri_intri conversion
    - Both output world2cam (w2c) format (compatible with VGGT)
    
    Usage:
        >>> config = DA3SpatialEncoderConfig(model_name="da3-large")
        >>> encoder = DA3SpatialEncoderPreTrainedModel(config)
        >>> encoder.load_pretrained_weights("depth-anything/DA3-LARGE-1.1")
        >>> _, _, camera_encs = encoder(video_tensor, return_cam_enc=True, grid_thw=video_grid_thw)
    """
    
    config_class = DA3SpatialEncoderConfig
    base_model_prefix = "spatial_encoder"
    
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False
    
    def __init__(self, config: DA3SpatialEncoderConfig):
        super().__init__(config)
        
        if not DA3_AVAILABLE:
            error_msg = (
                f"DA3 is not available. Import error: {DA3_IMPORT_ERROR if 'DA3_IMPORT_ERROR' in globals() else 'Unknown'}\n"
                f"Please check:\n"
                f"1. Submodule exists: {DA3_PATH}\n"
                f"2. Run: export PYTHONPATH=$PYTHONPATH:{DA3_PATH}\n"
                f"3. Or install: pip install -e {DA3_PATH.parent}"
            )
            raise ImportError(error_msg)
        
        self.config = config
        
        # Initialize DA3 model
        print(f"[DA3] Initializing {config.model_name}...")
        
        if DA3_API_AVAILABLE:
            # Use API wrapper if available (handles HF Hub loading)
            self.da3_wrapper = DepthAnything3(model_name=config.model_name)
            self.da3_model = self.da3_wrapper.model  # Extract underlying DepthAnything3Net
        else:
            # Fallback: Manual initialization using config
            print("[DA3] API wrapper not available, using manual initialization...")
            model_config = load_config(MODEL_REGISTRY[config.model_name])
            self.da3_model = create_object(model_config)
            self.da3_wrapper = None
        
        self.da3_model.eval()
        
        # DA3 configuration
        self.use_ray_pose = config.use_ray_pose
        self.ref_view_strategy = config.ref_view_strategy
        self.process_res = config.process_res
        self.process_res_method = config.process_res_method
        
        print(f"[DA3] Configuration:")
        print(f"  - model_name: {config.model_name}")
        print(f"  - use_ray_pose: {self.use_ray_pose}")
        print(f"  - ref_view_strategy: {self.ref_view_strategy}")
        print(f"  - process_res: {self.process_res} (VGGT uses 518)")
        print(f"  - process_res_method: {self.process_res_method}")
        
    def _init_weights(self, module):
        """No custom weight initialization needed."""
        pass
    
    def load_pretrained_weights(self, pretrained_weight: str):
        """
        Load DA3 pretrained weights from Hugging Face Hub.
        
        Args:
            pretrained_weight: HuggingFace model name
                Examples: "depth-anything/DA3-LARGE-1.1", "depth-anything/DA3-BASE"
        """
        print(f"[DA3] Loading pretrained weights from: {pretrained_weight}")
        if DA3_API_AVAILABLE:
            self.da3_wrapper = DepthAnything3(model_name=pretrained_weight)
            self.da3_model = self.da3_wrapper.model
        else:
            # Fallback to manual loading (requires implementing weight loading logic)
            raise NotImplementedError("Pretrained weight loading requires DA3 API wrapper")
        self.da3_model.eval()
        print(f"[DA3] ✅ Model loaded successfully!")
        
    def preprocess_video_tensors(self, video_tensor: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Preprocess video tensors (compatibility method).
        
        Args:
            video_tensor: List of video tensors
            
        Returns:
            Unmodified video tensors (DA3 handles preprocessing internally)
        """
        return video_tensor
    
    def forward(
        self,
        video_tensor: List[torch.Tensor],
        return_cam_enc: bool = False,
        grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[Optional[List], Optional[int], Optional[List]]:
        """
        Forward pass - extract camera poses from video frames.
        
        Args:
            video_tensor: List of video tensors [S, 3, H, W] where S is num frames
            return_cam_enc: If True, return camera encodings for pose rope
            grid_thw: Grid dimensions [B, T, H, W] - optional
            
        Returns:
            Tuple of (spatial_embeds_list, patch_start_idx, camera_encs)
            - spatial_embeds_list: None (not used with DA3)
            - patch_start_idx: None (not used with DA3)
            - camera_encs: List [(None, None, None, [extrinsics_w2c, intrinsics])]
                extrinsics_w2c: [B, S, 3, 4] world-to-camera transformation
                intrinsics: [B, S, 3, 3] camera intrinsic matrix
        
        Note:
            DA3 outputs world2cam (w2c) format, same as VGGT.
            Images are automatically resized to process_res to be divisible by patch_size=14.
        """
        # Extract batch info from grid_thw
        if grid_thw is not None:
            T, H, W = grid_thw[0]  # grid_thw format is [T, H, W], not [B, T, H, W]
            B = 1  # Always single batch in our use case
        else:
            B = 1
            
        # Prepare input: video_tensor is list of [S, 3, H, W]
        # DA3 expects [B, N, 3, H, W]
        images = video_tensor[0].unsqueeze(0)  # [1, S, 3, H, W]
        S = images.shape[1]
        
        # JJ: Resize images to be divisible by patch_size=14
        _, _, H_orig, W_orig = images.shape[1:]
        PATCH_SIZE = 14
        
        # Check if resize is needed
        if H_orig % PATCH_SIZE != 0 or W_orig % PATCH_SIZE != 0:
            # Calculate target resolution
            if self.config.process_res is not None:
                # Use configured target resolution
                target_res = self.config.process_res
            else:
                # Round up to next multiple of PATCH_SIZE
                target_res = ((max(H_orig, W_orig) + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
            
            # Resize using torch interpolate (simpler than InputProcessor)
            # [1, S, 3, H, W] -> [S, 3, H, W] -> resize -> [S, 3, H', W'] -> [1, S, 3, H', W']
            images_flat = images.squeeze(0)  # [S, 3, H, W]
            images_resized = torch.nn.functional.interpolate(
                images_flat,
                size=(target_res, target_res),
                mode='bilinear',
                align_corners=False
            )  # [S, 3, H', W']
            images = images_resized.unsqueeze(0)  # [1, S, 3, H', W']
            
            print(f"[DA3] Resized images from {H_orig}x{W_orig} to {images.shape[-2]}x{images.shape[-1]}")
        
        # JJ: Important - DA3 expects images in [0, 1] range
        # Check if normalization is needed
        if images.max() > 1.0:
            images = images / 255.0
        
        # Run DA3 forward (use the low-level forward, not inference API)
        with torch.no_grad():
            # Determine autocast dtype
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
                output = self.da3_model(  # JJ: Direct call to DepthAnything3Net
                    x=images,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],  # JJ: Use empty list, not None
                    infer_gs=False,
                    use_ray_pose=self.use_ray_pose,
                    ref_view_strategy=self.ref_view_strategy,
                )
        
        # Extract poses from DA3 output
        # ⚠️ CRITICAL: DA3 has inconsistent output formats!
        # Based on source code analysis (da3.py):
        # - Line 225: cam_dec: c2w → affine_inverse(c2w) → outputs w2c ✅
        # - Line 192: ray_head: w2c → affine_inverse(w2c) → outputs c2w ❌ (BUG!)
        extrinsics_raw = output['extrinsics']  # [B, S, 3, 4]
        intrinsics = output['intrinsics']      # [B, S, 3, 3]
        
        # JJ: Convert to consistent w2c format
        if self.use_ray_pose:
            # Ray head outputs c2w, need to convert to w2c
            # affine_inverse requires [B, S, 4, 4], so we need to add homogeneous row
            from depth_anything_3.utils.geometry import affine_inverse
            
            # Add homogeneous row [0, 0, 0, 1]
            extrinsics_c2w_homo = torch.zeros(B, S, 4, 4, device=extrinsics_raw.device, dtype=extrinsics_raw.dtype)
            extrinsics_c2w_homo[:, :, :3, :] = extrinsics_raw
            extrinsics_c2w_homo[:, :, 3, 3] = 1.0
            
            # c2w -> w2c
            extrinsics_w2c_homo = affine_inverse(extrinsics_c2w_homo)  # [B, S, 4, 4]
            extrinsics_w2c = extrinsics_w2c_homo[:, :, :3, :]  # [B, S, 3, 4]
            
            print("[DA3-WARNING] Ray head outputs c2w, converted to w2c for consistency")
        else:
            # Cam dec outputs w2c (correct format)
            extrinsics_w2c = extrinsics_raw
            print("[DA3-INFO] Camera decoder outputs w2c (no conversion needed)")
        
        # Sanity check
        assert extrinsics_w2c.shape == (B, S, 3, 4), \
            f"Expected extrinsics shape [B={B}, S={S}, 3, 4], got {extrinsics_w2c.shape}"
        assert intrinsics.shape == (B, S, 3, 3), \
            f"Expected intrinsics shape [B={B}, S={S}, 3, 3], got {intrinsics.shape}"
        
        if not return_cam_enc:
            # For connector (we don't use connector with DA3)
            return None, None
        else:
            # For pose rope: pack into camera_encs format compatible with VGGT
            # Format: [(None, None, None, [extrinsics, intrinsics])]
            # This matches the expected format in custom_spatial_mllm_pose_rope.py
            camera_encs = [(None, None, None, [extrinsics_w2c, intrinsics])]
            return None, None, camera_encs


# JJ: Verification function
def verify_da3_installation():
    """Verify DA3 is properly installed and accessible."""
    try:
        from depth_anything_3.model.da3 import DepthAnything3Net
        print("[DA3] ✅ Installation verified!")
        return True
    except ImportError as e:
        print(f"[DA3] ❌ Import failed: {e}")
        print(f"[DA3] Please check: {DA3_PATH}")
        return False


if __name__ == "__main__":
    # Test DA3 availability
    verify_da3_installation()
