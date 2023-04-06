import cv2
import random
import warnings
import numpy as np
from pathlib import Path
import torch
import mmcv
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.builder import DATASETS

from .base import BaseDataset


@PIPELINES.register_module()
class TransformLVISAnnotations:
    """
    Pipeline for transforming LVIS dataset annotations.
    """

    def __init__(self, max_overlap_ratio=0.5):
        """
        Constructor method for TransformLVISAnnotations pipeline.

        Args:
            max_overlap_ratio (float): Maximum overlap ratio to use for filtering.
        """
        self.max_overlap_ratio = max_overlap_ratio

    def __call__(self, results):
        """
        Method to apply the pipeline to input results.

        Args:
            results (dict): Dictionary containing the input results.

        Returns:
            dict: Dictionary containing the transformed results.
        """
        # Get size of input image.
        size = results['img_shape'][:2]

        # Create ground truth semantic segmentation array.
        gt_semantic_seg = np.zeros(size, dtype=np.uint8)

        # Create a list of segment information.
        segments_info = list()

        # Remove image meta and annotation meta information from input dictionary.
        results.pop('img_meta_info')
        ann_info = results.pop('ann_meta_info')

        # Shuffle the annotations.
        random.shuffle(ann_info)

        # Create a dictionary to store the areas of each annotation.
        areas = dict()

        # Loop through the annotations.
        for idx, info in enumerate(ann_info, 1):
            # Convert polygons to binary mask.
            mask = self.polygons2mask(info['segmentation'], size)

            # Calculate area of mask.
            area = mask.astype(float).sum()

            # If area is too small, skip the annotation.
            if area < torch.finfo(torch.float).eps:
                continue

            # Convert mask to boolean array.
            mask = mask.astype(bool)

            # Calculate overlap areas.
            overlap_areas = np.bincount(gt_semantic_seg[mask].flatten())

            # Calculate overlap ratio.
            overlap_ratio = overlap_areas[1:].sum() / area

            # Calculate maximum overlap ratio.
            overlap_ratio = max([overlap_ratio] + [
                overlap_areas[i] / areas[i] for i in areas
                if i < len(overlap_areas)])

            # If overlap ratio is too high, skip the annotation.
            if overlap_ratio > self.max_overlap_ratio:
                continue

            # Add area to areas dictionary.
            areas[idx] = area

            # Set annotation in ground truth semantic segmentation.
            gt_semantic_seg[mask] = idx

            # Add segment information to list.
            segments_info.append(dict(id=idx))

        # Add ground truth semantic segmentation to output dictionary.
        results['gt_semantic_seg'] = gt_semantic_seg

        # Add semantic segmentation field to output dictionary.
        results['seg_fields'].append('gt_semantic_seg')

        # Add segments information to output dictionary.
        results['segments_info'] = segments_info

        return results

    @staticmethod
    def polygons2mask(polygons, size):
        """
        Static method to convert polygons to binary mask.

        Args:
            polygons (list): List of polygons.
            size (tuple): Size of the image.

        Returns:
            np.ndarray: Numpy array containing the binary mask.
        """
        mask = np.zeros(size, dtype=np.uint8)
        for polygon in polygons:
            points = np.array(polygon).reshape((-1, 2))
            points = np.round(points).astype(np.int32)
            cv2.fillPoly(mask, points[None, ...], 1)
        return mask

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'max_overlap_ratio={self.max_overlap_ratio})'


@DATASETS.register_module()
class LVISDataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 data_root,
                 train=True,
                 ignore_prefix_file=None,
                 **kwargs):

        # determine the mode of the dataset
        self.mode = 'train' if train else 'val'

        # set of prefixes to ignore
        ignore_prefix = set()

        # if ignore prefix file is given, load prefixes to ignore
        if ignore_prefix_file is not None:
            if Path(ignore_prefix_file).is_file():
                ignore_prefix = mmcv.load(ignore_prefix_file)
                # set ignore_prefix to mode-specific prefixes if available
                if self.mode in ignore_prefix:
                    ignore_prefix = set(ignore_prefix[self.mode])
                # if mode-specific prefixes are not available, issue a warning
                else:
                    warnings.warn(
                        f'not found {self.mode} ignore_prefix in {ignore_prefix_file}')
            # if ignore prefix file is not found, issue a warning
            else:
                warnings.warn(
                    f'not found ignore_prefix_file {ignore_prefix_file}')

        # call the constructor of the BaseDataset class
        super(LVISDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            img_suffix='.jpg',
            ann_suffix=None,
            ignore_prefix=ignore_prefix,
            ignore_index=255,
            reduce_zero_label=False,
            gt_seg_map_loader_cfg=[dict(type='TransformLVISAnnotations')])

    def load_annotations(self):
        """
        Load the annotations for the dataset.
        Returns:
            img_infos (list): A list of dictionaries containing image file names and associated metadata.
        """
        root = self.data_root

        # Get the file name by the prefix of the image
        img_files = {int(p.stem): p for p in
                     (root / f'{self.mode}2017').rglob('*' + self.img_suffix)
                     if p.stem not in self.ignore_prefix}

        # Get the annotation file (with '.json' as the suffix)
        # Load the annotations from the annotation file
        infos = mmcv.load(next(root.rglob(f'lvis_v1_{self.mode}.json')))
        ann_infos = dict()

        for info in infos['annotations']:
            if info['image_id'] in img_files:
                ann_infos.setdefault(info['image_id'], list()).append(info)

        img_infos = list()

        for info in infos['images']:
            if info['id'] in img_files and info['id'] in ann_infos:
                img_infos.append(dict(
                    img_file=img_files[info['id']],
                    ann_meta_info=ann_infos[info['id']],
                    img_meta_info=info))

        return img_infos
