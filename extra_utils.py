from gym.spaces import Box
import numpy as np

def is_image_space(space: Box, check_channels_last: bool = True) -> bool:
    """
    Check if a Box space is an image space.
    It must have dtype uint8 and shape (H, W, C) if channels_last, else (C, H, W).
    """
    if not isinstance(space, Box) or space.dtype != np.uint8:
        return False

    if check_channels_last:
        return len(space.shape) == 3 and space.shape[2] in [1, 3]
    else:
        return len(space.shape) == 3 and space.shape[0] in [1, 3]
