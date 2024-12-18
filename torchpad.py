import torch
import torch.nn as nn
from math import ceil


def tpad(inpt, target):
    """
    Pad or crop input tensor to match target size, with flexibility to handle arbitrary dimensions
    and skip specific dimensions by using -1 in the target.

    Args:
        inpt (torch.Tensor): Input tensor to be padded or cropped, of shape (B, C, *dims).
        target (torch.Tensor, tuple of int, or int): 
            Target tensor or target dimensions. If a tensor, its shape determines the target size.
            If a tuple, dimensions specify the target size per dimension.
            If an int, all spatial dimensions are set to the same size.

    Returns:
        torch.Tensor: Resized (padded or cropped) input tensor matching the size of the target.
    """
    # Determine target dimensions
    if torch.is_tensor(target):
        target_shape = target.shape[2:]
    elif isinstance(target, tuple):
        target_shape = target
    elif isinstance(target, int):
        # Apply to all spatial dimensions
        target_shape = (target,) * (inpt.ndim - 2)
    else:
        raise ValueError(
            "Invalid target type. Must be a tensor, tuple, or int.")

    # Validate target shape length matches spatial dimensions
    if len(target_shape) != (inpt.ndim - 2):
        raise ValueError(
            "Target dimensions must match the spatial dimensions of the input.")

    # Compute the padding and cropping for each dimension
    delta = [
        target_shape[i] - inpt.shape[2 + i] if target_shape[i] != -1 else 0
        for i in range(len(target_shape))
    ]

    pad = []
    slices = [slice(None), slice(None)]  # For batch and channel dimensions

    for i, d in enumerate(delta):
        if d > 0:  # Padding required
            pad.extend([ceil(d / 2), d - ceil(d / 2)])
            slices.append(slice(None))  # No cropping needed
        elif d < 0:  # Cropping required
            pad.extend([0, 0])  # No padding
            slices.append(slice(-d // 2, -d // 2 + target_shape[i]))
        else:  # No padding or cropping
            pad.extend([0, 0])
            slices.append(slice(None))

    # Reverse pad order for nn.functional.pad (works rightmost dimension first)
    pad = pad[::-1]

    # Apply padding and cropping
    padded = nn.functional.pad(inpt, pad, mode='constant', value=0.0)
    resized = padded[tuple(slices)]

    return resized.to(dtype=inpt.dtype, device=inpt.device)
