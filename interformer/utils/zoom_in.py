from typing import Union
import numpy as np
import torch


def get_label_ids_with_areas(mask):
    if isinstance(mask, np.ndarray):
        if not mask.any():
            return None, None
        areas = np.bincount(mask.astype(np.uint8).flat)
        label_ids = np.nonzero(areas)[0]
        label_ids = list(filter(lambda x: x != 0, label_ids))
        return label_ids, areas[label_ids].tolist()
    elif isinstance(mask, torch.Tensor):
        # todo
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_bbox_from_mask(mask: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Returns the bounding boxes (left, up, right, bottom) of each mask in the given tensor.

    :param mask: A tensor of shape (*, height, width).
    :type mask: Union[torch.Tensor, np.ndarray]
    :return: A tensor of shape (*, 4).
    :rtype: Union[torch.Tensor, np.ndarray]
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')

    ori_shape = mask.shape
    to_tensor = isinstance(mask, torch.Tensor)

    rows, cols = mask.any(-1), mask.any(-2)
    rows = rows.detach().cpu().numpy() if to_tensor else rows
    rows = rows.reshape((-1, rows.shape[-1]))

    cols = cols.detach().cpu().numpy() if to_tensor else cols
    cols = cols.reshape((-1, cols.shape[-1]))

    results = []
    for row, col in zip(rows, cols):
        if not row.any() or not col.any():
            left, up, right, bottom = 0, 0, ori_shape[-1], ori_shape[-2]
        else:
            row, col = np.where(row)[0], np.where(col)[0]
            left, up, right, bottom = col[0], row[0], col[-1] + 1, row[-1] + 1
        results.append((left, up, right, bottom))

    bbox = np.array(results).reshape(ori_shape[:-2] + (4,))
    bbox = torch.from_numpy(bbox).to(mask.device) if to_tensor else bbox

    return bbox


def expand_bbox(bbox: Union[torch.Tensor, np.ndarray],
                h_ratio: float,
                w_ratio: float,
                height: int,
                width: int,
                h_min: int = 0,
                w_min: int = 0) -> Union[torch.Tensor, np.ndarray]:
    """
    Expand bounding box by given ratios and minimum sizes.

    :param bbox: A tensor or ndarray with shape (*, 4), each includes (left, up, right, bottom)
    :param h_ratio: The expand ratio of height
    :param w_ratio: The expand ratio of width
    :param height: The height of the image
    :param width: The width of the image
    :param h_min: The minimum height of the expanded bbox
    :param w_min: The minimum width of the expanded bbox
    :return: A tensor or ndarray with shape (*, 4), each includes (left, up, right, bottom)
    """

    # Check the type and shape of the input bbox
    if not isinstance(bbox, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Cannot handle type of bbox: {type(bbox)}")
    if bbox.shape[-1] != 4:
        raise ValueError(f"`bbox` is expected to have the shape: (*, 4), "
                         f"but got `bbox` of shape: {tuple(bbox.shape)}")

    # Convert bbox to a tensor and split it into 4 tensors
    ori_bbox = bbox
    to_array = isinstance(bbox, np.ndarray)
    bbox = torch.from_numpy(bbox) if to_array else bbox
    left, up, right, bottom = torch.chunk(bbox, 4, dim=-1)

    # Compute the center coordinates of the bbox
    xc, yc = (left + right) / 2.0, (up + bottom) / 2.0

    # Expand the bbox according to the ratios
    left = torch.round(xc - w_ratio * (xc - left)).clip(0, None)
    right = torch.round(xc - w_ratio * (xc - right)).clip(None, width)
    up = torch.round(yc - h_ratio * (yc - up)).clip(0, None)
    bottom = torch.round(yc - h_ratio * (yc - bottom)).clip(None, height)

    # Apply minimum size constraints to the expanded bbox
    if w_min > 0:
        _left = torch.round(xc - w_min / 2.0).clip(0, None)
        left = torch.where(left < _left, left, _left)
        _right = torch.round(xc + w_min / 2.0).clip(None, width)
        right = torch.where(right > _right, right, _right)

    if h_min > 0:
        _up = torch.round(yc - h_min / 2.0).clip(0, None)
        up = torch.where(up < _up, up, _up)
        _bottom = torch.round(yc + h_min / 2.0).clip(None, height)
        bottom = torch.where(bottom > _bottom, bottom, _bottom)

    # Concatenate the expanded bbox tensors and convert it back to the original type
    bbox = torch.cat([left, up, right, bottom], dim=-1).to(bbox)
    bbox = bbox.numpy() if to_array else bbox.to(ori_bbox)
    return bbox
