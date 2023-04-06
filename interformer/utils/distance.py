from typing import Union
import cv2
import numpy as np
import torch


def mask_to_distance(mask: Union[torch.Tensor, np.ndarray],
                     boundary_padding: bool) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a binary mask to a distance map.

    :param mask: A binary mask of shape (*, height, width).
    :param boundary_padding: A boolean flag indicating whether to pad the
    boundary of the mask before computing the distance map.
    :return: A distance map of the same shape as the input mask.
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')

    to_tensor = isinstance(mask, torch.Tensor)
    device = mask.device if to_tensor else None
    mask = mask.detach().cpu().numpy() if to_tensor else mask

    if boundary_padding:
        mask = np.pad(mask, [(0, 0)] * (len(mask.shape) - 2) + [(1, 1)] * 2)

    dist = np.stack(
        list(map(
            lambda x: cv2.distanceTransform(x, cv2.DIST_L2, 0),
            mask.reshape((-1, ) + mask.shape[-2:]).astype(np.uint8))),
        axis=0).reshape(mask.shape)

    if boundary_padding:
        dist = dist[..., 1:-1, 1:-1]

    dist = torch.from_numpy(dist).to(device) if to_tensor else dist
    return dist
