from typing import Union

import numpy as np
import torch


REF_DEFINITELY_BACKGROUND = 0
REF_POSSIBLY_BACKGROUND = 1
REF_UNKNOWN = 2
REF_POSSIBLY_FOREGROUND = 3
REF_DEFINITELY_FOREGROUND = 4
REF_MODES = [
    REF_DEFINITELY_BACKGROUND,
    REF_POSSIBLY_BACKGROUND,
    REF_UNKNOWN,
    REF_POSSIBLY_FOREGROUND,
    REF_DEFINITELY_FOREGROUND]
REF_INVERSE_MODES = {
    REF_POSSIBLY_BACKGROUND: REF_POSSIBLY_FOREGROUND,
    REF_POSSIBLY_FOREGROUND: REF_POSSIBLY_BACKGROUND}


def update_ref_label_with_mask(
    x: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    mode: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Update the reference label x with a binary mask

    :param x: Input array, shape (*, height, width)
    :param mask: Binary mask array, shape (*, height, width)
    :param mode: The mode of the reference label, must be one of `REF_MODES`
    :return: The updated reference label array, shape (*, height, width)
    """
    if not isinstance(x, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of x: {type(x)}')
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')
    if type(x) != type(mask):
        raise TypeError(f'Different types between x and mask: '
                        f'{type(x)} and {type(mask)}')
    if mode not in REF_MODES:
        raise ValueError(f'Unknown value of mode: {mode}')

    if isinstance(x, np.ndarray):
        res = x.copy()
        mask = mask.astype(bool)
        if mode == REF_UNKNOWN:
            return res
        if mode in [REF_DEFINITELY_BACKGROUND, REF_DEFINITELY_FOREGROUND]:
            res[mask] = mode
        else:
            inverse_mode = REF_INVERSE_MODES[mode]
            res[np.logical_and(mask, x == REF_UNKNOWN)] = mode
            res[np.logical_and(mask, x == inverse_mode)] = REF_UNKNOWN
    else:
        res = x.detach().clone()
        mask = mask.bool()
        if mode == REF_UNKNOWN:
            return res
        if mode in [REF_DEFINITELY_BACKGROUND, REF_DEFINITELY_FOREGROUND]:
            res[mask] = mode
        else:
            inverse_mode = REF_INVERSE_MODES[mode]
            res[torch.logical_and(mask, x == REF_UNKNOWN)] = mode
            res[torch.logical_and(mask, x == inverse_mode)] = REF_UNKNOWN
    return res


def update_ref_label(
    x: Union[torch.Tensor, np.ndarray],
    label: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Update the reference label x with a reference label

    :param x: Input array, shape (*, height, width)
    :param label: Reference label array, shape (*, height, width)
    :return: The updated reference label array, shape (*, height, width)
    """
    for mode in REF_MODES:
        x = update_ref_label_with_mask(x, label == mode, mode)
    return x
