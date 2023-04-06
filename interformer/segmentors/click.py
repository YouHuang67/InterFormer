import warnings
from typing import List, Tuple

import numpy as np
import torch

from mmseg.models.builder import SEGMENTORS
from mmseg.core.evaluation.metrics import eval_metrics
from interformer import utils
from .base import BaseInterSegmentor


@SEGMENTORS.register_module()
class ClickSegmentor(BaseInterSegmentor):

    @staticmethod
    def update_ref_label_by_point_lists(
            ref_label: torch.Tensor,
            point_lists: List[List[Tuple[int, int, str]]],
            inner_radius: int = 5,
            outer_radius: int = 0) -> torch.Tensor:
        """
        Update reference label by given point lists

        :param ref_label: shape (batch_size, 1, height, width)
        :param point_lists: a list of point lists,
                            with each point represented by (y, x, mode)
        :param inner_radius: int
        :param outer_radius: int
        :return: shape (batch_size, 1, height, width)
        """
        if ref_label.dim() != 4 or ref_label.size(1) != 1:
            raise ValueError(f"`ref_label` is expected to have a shape "
                             f"(batch_size, 1, height, width), "
                             f"but got shape {tuple(ref_label.size())}")

        for idx, points in enumerate(list(point_lists)):
            points = [
                (y, x, mode)
                for y, x, mode in points
                if y is not None and x is not None and mode is not None
            ]
            point_lists[idx] = points

        new_ref_label = utils.point_list_to_ref_labels(
            ref_label, point_lists, inner_radius, outer_radius)
        ref_label = utils.update_ref_label(ref_label, new_ref_label)
        return ref_label

    @staticmethod
    def update_ref_label_by_prediction(
            ref_label: torch.Tensor,
            pre_label: torch.Tensor) -> torch.Tensor:
        """
        Update reference label by given prediction

        :param ref_label: shape (batch_size, 1, height, width)
        :param pre_label: shape (batch_size, 1, height, width)
        :return: shape (batch_size, 1, height, width)
        """
        ref_label = utils.update_ref_label_with_mask(
            ref_label, pre_label == 1, utils.REF_POSSIBLY_FOREGROUND)
        ref_label = utils.update_ref_label_with_mask(
            ref_label, pre_label == 0, utils.REF_POSSIBLY_BACKGROUND)
        return ref_label

    @staticmethod
    def sample_num_clicks(max_num_clicks: int, gamma: float) -> int:
        """
        Sample the number of clicks

        :param max_num_clicks: int, the maximum number of clicks
        :param gamma: float, the decay rate of the probabilities
        :return: int, the number of clicks
        """
        probs = gamma ** np.arange(max_num_clicks + 1)
        probs /= probs.sum()
        return np.random.choice(range(len(probs)), p=probs)

    @classmethod
    def click(
            cls,
            ref_label: torch.Tensor,
            pre_label: torch.Tensor,
            seg_label: torch.Tensor,
            point_lists: List[List[Tuple[int, int, str]]],
            inner_radius: int,
            outer_radius: int,
            sfc_inner_k: float,
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int, str]]]]:
        """
        Given a reference label, a predicted label, a segmentation label, a list of
        points, and radii, return the updated reference label and point lists.

        :param ref_label: shape (batch_size, 1, height, width)
        :param pre_label: shape (batch_size, 1, height, width)
        :param seg_label: shape (batch_size, 1, height, width)
        :param point_lists: a list of point lists, with each point represented by
                             (y, x, mode)
        :param inner_radius: the inner radius of the updated reference label
        :param outer_radius: the outer radius of the updated reference label
        :param sfc_inner_k: the scaling factor of the inner radius
        :return: a tuple containing the updated reference label and point lists
        """

        # If the segmentation label has multiple values, transform it into a binary mask
        if torch.unique(seg_label).nelement() > 2:
            warnings.warn(f'found multiple values of `seg_label` '
                          f'{torch.unique(seg_label).cpu().numpy().tolist()}, '
                          f'transform `seg_label` into binary mask')
            seg_label = (seg_label > 0).to(seg_label)

        new_point_lists = list()

        # Iterate through the point lists and update them
        for idx, points in enumerate(list(point_lists)):
            y, x, mode = utils.click(
                pre_label=pre_label[idx, 0],
                seg_label=seg_label[idx, 0],
                points=points,
                sfc_inner_k=sfc_inner_k
            )

            # If the updated point is not None, add it to the new point list and
            # update the original point list
            if y is not None and x is not None and mode is not None:
                new_point_lists.append([(y, x, mode)])
                point_lists[idx].extend(new_point_lists[-1])
            else:
                new_point_lists.append(list())

        # Update the reference label by the updated point lists
        ref_label = cls.update_ref_label_by_point_lists(
            ref_label=ref_label,
            point_lists=new_point_lists,
            inner_radius=inner_radius,
            outer_radius=outer_radius
        )

        return ref_label, point_lists

    @torch.no_grad()
    def interact_train(
            self,
            img: torch.Tensor,
            img_metas: torch.Tensor,
            gt_semantic_seg: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # Check if gt_semantic_seg is multi-label
        if gt_semantic_seg.size(1) != 1:
            raise ValueError(f'Cannot handle multi `gt_semantic_seg` '
                             f'with shape {tuple(gt_semantic_seg.size())}')

        # Get training configuration
        cfg = self.train_cfg

        # Set points configuration
        pts_cfg = dict(
            inner_radius=cfg.inner_radius,
            outer_radius=cfg.outer_radius
        )

        # Initialize empty point lists and reference label tensor
        point_lists = [list() for _ in range(img.size(0))]
        ref_label = torch.ones_like(gt_semantic_seg) * utils.REF_UNKNOWN

        # First click
        ref_label, point_lists = self.click(
            ref_label=ref_label,
            pre_label=torch.zeros_like(gt_semantic_seg),
            seg_label=(gt_semantic_seg == 1).to(gt_semantic_seg),
            point_lists=point_lists,
            **pts_cfg,
            sfc_inner_k=1.0
        )

        # Sample the number of clicks
        num_clicks = self.sample_num_clicks(cfg.max_num_clicks, cfg.gamma)

        # Backbone forward pass if num_clicks > 0
        x = self.backbone(img) if num_clicks > 0 else None

        # Iterate over num_clicks and update reference label tensor and point lists
        for _ in range(num_clicks):
            out = self.encode_decode(img, None, x=x, ref_label=ref_label)
            out = out.argmax(dim=1, keepdim=True)
            ref_label = self.update_ref_label_by_prediction(ref_label, out)
            ref_label, point_lists = self.click(
                ref_label=ref_label,
                pre_label=out,
                seg_label=(gt_semantic_seg == 1).to(gt_semantic_seg),
                point_lists=point_lists,
                **pts_cfg,
                sfc_inner_k=cfg.sfc_inner_k
            )

        # Return ground truth and reference label tensor
        return gt_semantic_seg, dict(ref_label=ref_label)

    @torch.no_grad()
    def interact_test(
            self,
            img: torch.Tensor,
            img_metas: torch.Tensor,
            gt_semantic_seg: torch.Tensor) -> List[float]:
        """
        Performs interactive testing on a single image.

        Args:
            img (torch.Tensor): input image tensor with shape (1, C, H, W).
            img_metas (list[dict]): list of image meta information dictionaries.
            gt_semantic_seg (torch.Tensor): ground truth semantic segmentation
                tensor with shape (1, 1, H, W).

        Returns:
            list[float]: list of IoU scores obtained after each click.
        """
        cfg = self.test_cfg

        # Define point selection configuration.
        pts_cfg = dict(
            inner_radius=cfg.inner_radius,
            outer_radius=cfg.outer_radius
        )

        # Check if only one sample is present in the batch.
        if img.size(0) > 1:
            raise ValueError(f'Only a single sample per batch is allowed, '
                             f'but got {img.size(0)} samples in this batch')

        # Check if gt_semantic_seg has the correct shape.
        if gt_semantic_seg.dim() != 4 or \
                gt_semantic_seg[..., 0, 0].nelement() > 1:
            raise ValueError(f'`gt_semantic_seg` is expected to have '
                             f'shape (1, 1, height, width), but '
                             f'got shape {tuple(gt_semantic_seg.size())}')

        # Initialize points and results.
        points, results = list(), list()

        # Initialize reference label.
        ref_label = torch.ones_like(gt_semantic_seg) * utils.REF_UNKNOWN

        # Compute feature map and segmentation map from the input image.
        z, out = self.backbone(img), torch.zeros_like(gt_semantic_seg)

        # Perform interactive testing.
        for _ in range(cfg.num_clicks):
            # Perform point selection.
            ref_label, (points, ) = self.click(
                ref_label=ref_label,
                pre_label=out,
                seg_label=(gt_semantic_seg == 1).to(gt_semantic_seg),
                point_lists=[points],
                **pts_cfg,
                sfc_inner_k=1.0
            )

            # Compute segmentation map from the selected points.
            out = self.encode_decode(
                img, img_metas, x=z, ref_label=ref_label
            )
            out = out.argmax(dim=1, keepdim=True)

            # Compute and record the IoU score.
            results.append(eval_metrics(
                list(out.detach().cpu().numpy()),
                list(gt_semantic_seg.cpu().numpy()),
                num_classes=2,
                ignore_index=(
                    cfg.ignore_index if hasattr(cfg, 'ignore_index') else 255
                ),
                metrics=['mIoU'],
                reduce_zero_label=False
            )['IoU'][1])

            # Update reference label using the latest segmentation map.
            ref_label = self.update_ref_label_by_prediction(ref_label, out)

        return results
