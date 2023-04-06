import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class BinaryIoU(nn.Module):

    def __init__(self,
                 ignore_index=255,
                 threshold=0.0,
                 loss_name='binary_iou',
                 **kwargs):
        super(BinaryIoU, self).__init__()
        self.ignore_index = ignore_index
        self.threshold = threshold
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        def format_pred(x, dim):
            if x.dim() == dim + 1:
                if x.size(1) == 1:
                    x = x.squeeze(1)
                elif x.size(1) == 2:
                    x = x[:, 1] - x[:, 0]
            assert x.dim() == dim, \
                f'cannot format x with shape: {x.size()} ' \
                f'to match the dim {dim}'
            return x

        if target.dim() == 2:
            target_dim = 2
        elif target.dim() in [3, 4]:
            target_dim = 3
            if target.dim() == 4:
                assert target.size(1) == 1, \
                    f'cannot handle target with shape: {target.size()}'
                target = target.squeeze(1)
        else:
            raise NotImplementedError(
                f'cannot handle target with shape: {target.size()}')
        pred = format_pred(pred, target_dim)

        ious = list()
        for pred_label, label in zip(pred > self.threshold, target):
            mask = (label != self.ignore_index)
            pred_label = pred_label[mask].float()
            label = label[mask].float()
            intersect = pred_label[pred_label == label]
            area_intersect = torch.histc(intersect, bins=2, min=0, max=1)[1]
            area_pred_label = torch.histc(pred_label, bins=2, min=0, max=1)[1]
            area_label = torch.histc(label, bins=2, min=0, max=1)[1]
            area_union = area_pred_label + area_label - area_intersect
            ious.append(torch.nan_to_num(area_intersect / area_union))
        return torch.stack(ious)

    @property
    def loss_name(self):
        return self._loss_name
