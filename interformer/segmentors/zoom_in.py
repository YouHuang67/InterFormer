from typing import List, Tuple
import torch
from mmseg.models.builder import SEGMENTORS
from mmseg.core.evaluation.metrics import eval_metrics

from .. import utils
from .click import ClickSegmentor


@SEGMENTORS.register_module()
class ClickSegmentorZoomIn(ClickSegmentor):

    @staticmethod
    def get_roi(pre_label: torch.Tensor,
                ref_label: torch.Tensor,
                expand_ratio: Tuple[float],
                max_stride: int,
                min_size: int) -> torch.Tensor:
        """Get region of interest from pre_label and ref_label"""
        mask = (pre_label == 1) | \
               (ref_label == utils.REF_DEFINITELY_FOREGROUND) | \
               (ref_label == utils.REF_DEFINITELY_BACKGROUND)
        roi = utils.get_bbox_from_mask(mask)
        roi = utils.expand_bbox(
            roi,
            *expand_ratio,
            *pre_label.size()[-2:],
            min_size,
            min_size)
        left, up, right, bottom = torch.chunk(roi, 4, dim=-1)
        left = torch.floor(left / max_stride) * max_stride
        right = torch.ceil(right / max_stride) * max_stride
        up = torch.floor(up / max_stride) * max_stride
        bottom = torch.ceil(bottom / max_stride) * max_stride
        roi = torch.cat([left, up, right, bottom], dim=-1)
        return roi

    @torch.no_grad()
    def interact_test(self, img: torch.Tensor, img_metas: List[dict],
                      gt_semantic_seg: torch.Tensor) -> List[float]:
        """Perform interactive testing on a single image"""
        cfg = self.test_cfg
        pts_cfg = dict(inner_radius=cfg.inner_radius,
                       outer_radius=cfg.outer_radius)
        if img.size(0) > 1:
            raise ValueError(f'Only a single sample per batch is allowed, '
                             f'but got {img.size(0)} samples in this batch')
        if gt_semantic_seg.dim() != 4 or \
                gt_semantic_seg[..., 0, 0].nelement() > 1:
            raise ValueError(f'`gt_semantic_seg` is expected to have '
                             f'shape (1, 1, height, width), but '
                             f'got shape {tuple(gt_semantic_seg.size())}')

        points, results = list(), list()
        ref_label = torch.ones_like(gt_semantic_seg) * utils.REF_UNKNOWN
        x, out = self.backbone(img), torch.zeros_like(gt_semantic_seg)
        if hasattr(cfg, 'zoom_in_start_step'):
            zoom_in_start_step = cfg.zoom_in_start_step
        else:
            zoom_in_start_step = 1
        for step in range(cfg.num_clicks):
            ref_label, (points,) = self.click(
                ref_label=ref_label,
                pre_label=out,
                seg_label=gt_semantic_seg,
                point_lists=[points],
                **pts_cfg,
                sfc_inner_k=1.0)
            if step < zoom_in_start_step:
                roi = None
            else:
                roi = self.get_roi(
                    pre_label=out,
                    ref_label=ref_label,
                    expand_ratio=(cfg.expand_ratio, cfg.expand_ratio),
                    max_stride=max(img.size(-1) // i.size(-1) for i in x),
                    min_size=cfg.min_size)
            out = self.encode_decode(
                img, img_metas, x=x, ref_label=ref_label, roi=roi)
            out = out.argmax(dim=1, keepdim=True)
            results.append(eval_metrics(
                list(out.detach().cpu().numpy()),
                list(gt_semantic_seg.cpu().numpy()),
                num_classes=2,
                ignore_index=(cfg.ignore_index
                              if hasattr(cfg, 'ignore_index') else 255),
                metrics=['mIoU'],
                reduce_zero_label=False)['IoU'][1])
            ref_label = self.update_ref_label_by_prediction(ref_label, out)
        return results
