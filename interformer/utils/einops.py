__all__ = ['rearrange']
from einops import rearrange as _rearrange


def rearrange(*args, **kwargs):
    """
    Rearrange tensor axes according to a given pattern.

    Args:
        *args: A variable-length argument list of tensors to be rearranged.
        **kwargs: A variable-length keyword argument list of options for the rearrangement.

    Returns:
        The rearranged tensor.
    """
    return _rearrange(*args, **kwargs).contiguous()
