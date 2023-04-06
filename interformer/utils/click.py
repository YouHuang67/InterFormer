import random
from typing import Union, Optional, List, Tuple
import numpy as np
import mmcv
import torch

from .uncertainty import *
from .distance import mask_to_distance

CLK_POSITIVE = 'positive'
CLK_POSITIVE_NUM = 253
CLK_NEGATIVE = 'negative'
CLK_NEGATIVE_NUM = 254
CLK_MODES = [CLK_POSITIVE, CLK_NEGATIVE]
CLK_MODE_NUMS = [CLK_POSITIVE_NUM, CLK_NEGATIVE_NUM]


def points_to_ref_label(
        label: Union[torch.Tensor, np.ndarray],
        points: List[Tuple[int, int, str]],
        inner_radius: float,
        outer_radius: float
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert the clicked points to a reference label.

    :param label: A binary mask of shape (height, width) or (1, height, width).
    :param points: A list of (y, x, mode), where y and x are the coordinates of a point, and mode is either 'positive'
    or 'negative'.
    :param inner_radius: The inner radius for the reference label.
    :param outer_radius: The outer radius for the reference label.
    :return: A binary mask of shape (height, width) or (1, height, width) representing the reference label.
    """

    # Check input validity.
    if not isinstance(label, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(label)}')
    if not mmcv.is_list_of(list(points), tuple):
        raise TypeError(f'Cannot handle type of points: {type(points)}')
    if len(label.shape) != 2 and not (len(label.shape) == 3 and label.shape[0] == 1):
        raise ValueError(f'Cannot handle label of shape: {tuple(label.shape)}')
    for y, x, mode in points:
        if isinstance(float(y), float) and isinstance(float(x), float) and mode in CLK_MODES:
            continue
        raise ValueError(f'Found invalid point {(y, x, mode)} among points: {points}')

    # Initialize label.
    ori_label = label
    to_tensor = isinstance(label, torch.Tensor)
    label = torch.ones_like(label.cpu()) if to_tensor else np.ones_like(label)
    label = label * REF_UNKNOWN

    # If there are no points, return the original label.
    if len(points) == 0:
        label = label.to(ori_label) if to_tensor else label.astype(ori_label.dtype)
        return label

    # Compute the inner and outer masks for each point.
    inner_masks = dict()
    outer_masks = dict()
    shape = tuple(label.shape[-2:])
    for y, x, mode in points:
        if mode == CLK_POSITIVE:
            inner, outer = REF_DEFINITELY_FOREGROUND, REF_POSSIBLY_FOREGROUND
        else:
            inner, outer = REF_DEFINITELY_BACKGROUND, REF_POSSIBLY_BACKGROUND
        inner_mask = inner_masks.setdefault(inner, np.ones(shape))
        outer_mask = outer_masks.setdefault(outer, np.ones(shape))
        inner_mask[y, x], outer_mask[y, x] = 0, 0

    # Compute the reference label using the inner and outer masks.
    for outer in list(outer_masks):
        dist = mask_to_distance(outer_masks[outer], False)
        mask = (dist <= inner_radius + outer_radius)
        mask = torch.from_numpy(mask) if to_tensor else mask
        label[mask.reshape(label.shape)] = outer
        outer_masks[outer] = mask
    if REF_POSSIBLY_FOREGROUND in outer_masks and \
       REF_POSSIBLY_BACKGROUND in outer_masks:
        mask = outer_masks[REF_POSSIBLY_FOREGROUND] & \
               outer_masks[REF_POSSIBLY_BACKGROUND]
        label[mask.reshape(label.shape)] = REF_UNKNOWN

    for inner in list(inner_masks):
        dist = mask_to_distance(inner_masks[inner], False)
        mask = (dist <= inner_radius)
        mask = torch.from_numpy(mask) if to_tensor else mask
        label[mask.reshape(label.shape)] = inner
    label = label.to(ori_label) if to_tensor else label.astype(ori_label.dtype)
    return label


def point_list_to_ref_labels(labels: Union[torch.Tensor, np.ndarray],
                             point_list: List[List[Tuple[int, int, str]]],
                             inner_radius: float,
                             outer_radius: float) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a list of point lists to reference labels.

    :param labels: A tensor of shape (*, height, width).
    :param point_list: A list of point lists, where each point list contains tuples
                       of (x, y, class) representing the coordinates and class label
                       of each point.
    :param inner_radius: The inner radius for creating reference labels.
    :param outer_radius: The outer radius for creating reference labels.
    :return: A tensor of shape (*, height, width).
    """

    # Check the type of mask
    if not isinstance(labels, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(labels)}')

    # Check the number of point lists
    if len(point_list) != len(labels[..., 0, 0].reshape((-1,))):
        raise ValueError(f'Number of point lists {len(point_list)} '
                         f'less or more than the number of labels '
                         f'{tuple(labels[..., 0, 0].shape)}')

    # Save the original labels
    ori_labels = labels

    # Convert the labels tensor to CPU and reshape it
    to_tensor = isinstance(labels, torch.Tensor)
    labels = labels.cpu() if to_tensor else labels
    labels = labels.reshape((-1, ) + labels.shape[-2:])

    # Create reference labels for each point list
    ref_labels = list()
    for idx, label in enumerate(labels):
        points = point_list[idx]
        ref_label = points_to_ref_label(label, points, inner_radius, outer_radius)
        ref_labels.append(ref_label)

    # Convert the reference labels tensor back to GPU and reshape it to the original shape
    if to_tensor:
        ref_labels = torch.stack(ref_labels, dim=0)
        ref_labels = ref_labels.view_as(ori_labels).to(ori_labels)
    else:
        ref_labels = np.stack(ref_labels, axis=0)
        ref_labels = ref_labels.reshape(ori_labels.shape)

    return ref_labels


def click(pre_label: Union[torch.Tensor, np.ndarray],
          seg_label: Union[torch.Tensor, np.ndarray],
          points: Optional[List[Tuple]] = None,
          sfc_inner_k: float = 1.0) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Click function for image segmentation

    :param pre_label: predicted label with shape (height, width)
    :param seg_label: ground truth label with shape (height, width)
    :param points: list of tuples (y, x, mode) representing clicked points
                   with mode being either 1 (positive) or 2 (negative)
    :param sfc_inner_k: float representing the adjustment factor for click area,
                        where 1.0 indicates the center of the click
    :return: tuple representing new click with format (y, x, mode)
    """
    # Check types and shapes of input labels
    if not isinstance(pre_label, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of pre_label: {type(pre_label)}')
    if not isinstance(seg_label, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of seg_label: {type(seg_label)}')
    if type(pre_label) != type(seg_label):
        raise TypeError(f'`pre_label` and `seg_label` are of different types: '
                        f'{type(pre_label)} and {type(seg_label)}')
    if tuple(pre_label.shape) != tuple(seg_label.shape):
        raise ValueError(f'`pre_label` and `seg_label` '
                         f'are of different shapes: '
                         f'{tuple(pre_label.shape)} and '
                         f'{tuple(seg_label.shape)}')
    if len(pre_label.shape) != 2:
        raise ValueError(f'Both `pre_label` and `seg_label` are expected to '
                         f'have the shape of (height, width), but got shape '
                         f'{tuple(pre_label.shape)}')

    # Check validity of input points
    if points is not None:
        height, width = seg_label.shape
        for y, x, mode in points:
            if (isinstance(float(y), float) and isinstance(float(x), float)) \
                    and (mode in CLK_MODES) and (0 <= y < height) and (0 <= x < width):
                continue
            raise ValueError(f'Found invalid point {(y, x, mode)} '
                             f'for {height}x{width} labels '
                             f'among points: {points}')

    # Calculate the distance scale based on sfc_inner_k
    if sfc_inner_k >= 1.0:
        dist_scale = 1 / (sfc_inner_k + torch.finfo(torch.float).eps)
    elif sfc_inner_k < 0.0:
        dist_scale = 0.0  # whole object area
    else:
        raise ValueError(f'Invalid sfc_inner_k: {sfc_inner_k}')

    # Convert labels to numpy arrays
    if isinstance(pre_label, torch.Tensor):
        pre_label = pre_label.detach().cpu().numpy()
        seg_label = seg_label.detach().cpu().numpy()
    pre_label = (pre_label == 1)
    seg_label = (seg_label == 1)

    # Create ignore mask based on points
    ignore_mask = np.zeros_like(pre_label, dtype=bool)
    for y, x, _ in (list() if points is None else points):
        ignore_mask[y, x] = True

    # Calculate distance maps based on ignore_mask
    fneg = np.logical_and(~pre_label, seg_label)
    fpos = np.logical_and(pre_label, ~seg_label)
    fneg = np.logical_and(fneg, ~ignore_mask)
    fpos = np.logical_and(fpos, ~ignore_mask)
    ndist = mask_to_distance(fneg, True)
    pdist = mask_to_distance(fpos, True)

    # Calculate maximum distances
    ndmax, pdmax = ndist.max(), pdist.max()
    if ndmax == pdmax == 0:
        return None, None, None

    # Determine click mode and points based on maximum distances
    if ndmax > pdmax:
        mode = CLK_POSITIVE
        points = np.argwhere(ndist > dist_scale * ndmax)
    else:
        mode = CLK_NEGATIVE
        points = np.argwhere(pdist > dist_scale * pdmax)

    if len(points) == 0:
        return None, None, None

    # Randomly choose a point from the points
    y, x = random.choice(points)
    return int(y), int(x), mode
