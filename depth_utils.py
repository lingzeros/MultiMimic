"""Shared preprocessing for raw single-channel RealSense depth frames."""

import numpy as np


# The inspected Peach episodes contain uint16 values around 404..1984. These
# defaults retain the useful workspace and suppress distant outliers. Keep the
# same values for training and deployment.
DEPTH_MIN_MM = 400.0
DEPTH_MAX_MM = 2000.0


def is_depth_name(name):
    return "depth" in str(name).lower()


def normalize_depth(depth, minimum_mm=DEPTH_MIN_MM, maximum_mm=DEPTH_MAX_MM):
    """Convert a raw HxW uint16 depth map to float32 [0, 1].

    Zero is treated as invalid and remains zero. Non-zero values are clipped to
    the configured millimetre range. The depth stays single-channel; callers
    add the leading channel dimension expected by PyTorch.
    """
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H,W] or [H,W,1], got {depth.shape}")
    if maximum_mm <= minimum_mm:
        raise ValueError("maximum_mm must be greater than minimum_mm")

    values = depth.astype(np.float32, copy=False)
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.float32)
    if np.any(valid):
        clipped = np.clip(values[valid], minimum_mm, maximum_mm)
        normalized[valid] = (clipped - minimum_mm) / (maximum_mm - minimum_mm)
    return normalized
