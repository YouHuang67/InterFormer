from typing import Union
import cv2
import numpy as np
import torch
from mmcv.utils.misc import to_2tuple


def erode(
    mask: Union[torch.Tensor, np.ndarray],
    kernel_size: int,
    iterations: int,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply erosion to the input mask.

    :param mask: A tensor or numpy array of shape (*, height, width).
    :param kernel_size: The size of the kernel used in erosion.
    :param iterations: The number of times erosion is applied.
    :return: A tensor or numpy array of the same shape as input mask.
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')

    to_tensor = isinstance(mask, torch.Tensor)
    device = mask.device if to_tensor else None
    mask = mask.detach().cpu().numpy() if to_tensor else mask
    kernel = np.ones(to_2tuple(kernel_size)).astype(np.uint8)
    mask = np.stack(
        list(map(
            lambda x: cv2.erode(x, kernel, iterations=iterations),
            mask.reshape((-1, ) + mask.shape[-2:]).astype(np.uint8))),
        axis=0).reshape(mask.shape)
    mask = torch.from_numpy(mask).to(device) if to_tensor else mask

    return mask


def dilate(
    mask: Union[torch.Tensor, np.ndarray],
    kernel_size: int,
    iterations: int,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply dilation to the input mask.

    :param mask: A tensor or numpy array of shape (*, height, width).
    :param kernel_size: The size of the kernel used in dilation.
    :param iterations: The number of times dilation is applied.
    :return: A tensor or numpy array of the same shape as input mask.
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')

    to_tensor = isinstance(mask, torch.Tensor)
    device = mask.device if to_tensor else None
    mask = mask.detach().cpu().numpy() if to_tensor else mask
    kernel = np.ones(to_2tuple(kernel_size)).astype(np.uint8)
    mask = np.stack(
        list(map(
            lambda x: cv2.dilate(x, kernel, iterations=iterations),
            mask.reshape((-1, ) + mask.shape[-2:]).astype(np.uint8))),
        axis=0).reshape(mask.shape)
    mask = torch.from_numpy(mask).to(device) if to_tensor else mask

    return mask
