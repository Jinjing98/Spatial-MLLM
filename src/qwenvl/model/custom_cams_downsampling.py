import torch
from typing import Tuple
def downsample_cams(intrisics,extrinsics_w2c, factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample the camera parameters by a factor of 2.
    Args:
        extrinsics_w2c: (B, S, 3, 4)
        intrisics: (B, S, 3, 3)
        factor: int
    Returns:
        intrisics_aligned: (B, S, 3, 3)
        extrinsics_aligned: (B, S, 3, 4)
    """
    intrisics_aligned = intrisics[:, ::factor]
    extrinsics_aligned = extrinsics_w2c[:, ::factor]
    return intrisics_aligned, extrinsics_aligned


