import torch.nn as nn
from mmcv.cnn import ACTIVATION_LAYERS


# Register a Hardswish activation function as a module with the mmcv library
@ACTIVATION_LAYERS.register_module()
class Hardswish(nn.Hardswish):
    """
    A Hardswish activation function that can be used as a module in a PyTorch network.
    """
    pass
