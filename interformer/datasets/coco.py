import warnings
import numpy as np
from pathlib import Path
from copy import deepcopy

import mmcv
from mmseg.datasets.builder import PIPELINES, DATASETS
from .base import BaseDataset


@PIPELINES.register_module()
class EncodeCOCOPanopticAnnotations:
    """
    Encode COCO Panoptic annotations by replacing RGB color encoding with
    segment IDs, and save annotations as a list of dictionaries.
    """
    def __call__(self, results):
        """
        Args:
            results (dict): A dictionary containing the following keys:
                - gt_semantic_seg (numpy.ndarray): Ground-truth semantic segmentation map encoded in RGB colors.
                - segments_info (list[dict]): A list of dictionaries containing the following keys:
                    * id (int): Segment ID.
                    * category_id (int): Category ID.
                    * iscrowd (bool): Whether the segment is crowd.
                - thing_category_ids (set): A set of category IDs for things (non-stuff) classes.

        Returns:
            dict: A dictionary containing the following keys:
                - gt_semantic_seg (numpy.ndarray): Ground-truth semantic segmentation map encoded in segment IDs.
                - segments_info (list[dict]): A list of dictionaries containing the following keys:
                    * id (int): Segment ID.
                    * isthing (int): Whether the segment is a thing (non-stuff) class (1) or not (0).
                    * iscrowd (bool): Whether the segment is crowd.
        """
        gt_semantic_seg = results.pop('gt_semantic_seg')
        segments_info = results.pop('segments_info')
        # Convert RGB color encoding to segment IDs
        gt_semantic_seg = (
                gt_semantic_seg[..., 0] * (1 << 16) +
                gt_semantic_seg[..., 1] * (1 << 8) +
                gt_semantic_seg[..., 2] * (1 << 0))
        results['gt_semantic_seg'] = np.zeros_like(gt_semantic_seg).astype(
            np.uint8)
        results['segments_info'] = list()
        thing_category_ids = results.pop('thing_category_ids')
        for i, info in enumerate(segments_info, 1):
            results['gt_semantic_seg'][gt_semantic_seg == info['id']] = i
            results['segments_info'].append(dict(
                id=i,
                isthing=int(info['category_id'] in thing_category_ids),
                iscrowd=info['iscrowd']))
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@DATASETS.register_module()
class COCOPanopticDataset(BaseDataset):
    """
    COCO Panoptic dataset.
    """
    def __init__(self,
                 pipeline,
                 data_root,
                 train=True,
                 ignore_prefix_file=None,
                 **kwargs):
        """
        Initialize the dataset.

        Args:
            pipeline (list[dict]): A list of dictionaries representing the image processing pipeline.
            data_root (str): Root directory where the dataset is stored.
            train (bool): Whether the dataset is for training or not.
            ignore_prefix_file (str): A file path to the file containing a list of prefixes for invalid samples that should be ignored.
            **kwargs: Other keyword arguments passed to the parent constructor.
        """
        self.mode = 'train' if train else 'val'
        ignore_prefix = set()
        if ignore_prefix_file is not None:
            if Path(ignore_prefix_file).is_file():
                ignore_prefix = mmcv.load(ignore_prefix_file)
                if self.mode in ignore_prefix:
                    ignore_prefix = set(ignore_prefix[self.mode])
                else:
                    warnings.warn(
                        f'not found {self.mode} ignore_prefix '
                        f'in {ignore_prefix_file}')
            else:
                warnings.warn(
                    f'not found ignore_prefix_file {ignore_prefix_file}')
        super(COCOPanopticDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            img_suffix='.jpg',
            ann_suffix='.png',
            ignore_prefix=ignore_prefix,
            ignore_index=255,
            reduce_zero_label=False,
            gt_seg_map_loader_cfg=[
                dict(type='LoadAnnotations'),
                dict(type='EncodeCOCOPanopticAnnotations')])

    def load_annotations(self):
        """Load annotation of COCO panoptic dataset."""
        data_root = self.data_root
        img_suffix = self.img_suffix
        ann_suffix = self.ann_suffix
        ignore_prefix = self.ignore_prefix

        # get the file name by the prefix of the image
        # (shared with its annotation file)
        img_files = {p.stem: p for p in data_root.rglob('*' + img_suffix)}
        ann_files = {p.stem: p for p in data_root.rglob('*' + ann_suffix)}

        img_infos = []
        # get the annotation file (with '.json' as the suffix)
        # load the annotations from the annotation file
        ann_infos = mmcv.load(list(data_root.rglob(
            '*' + f'panoptic_{self.mode}2017.json'))[0])
        # get all the category ids with isthing == 1
        thing_category_ids = set(
            x['id'] for x in ann_infos['categories'] if x['isthing'] == 1)
        for info in ann_infos['annotations']:
            prefix = Path(info.pop('file_name')).stem
            # filter invalid samples found before
            if prefix in ignore_prefix:
                continue
            # filter the samples lacking the corresponding image or annotation
            if prefix not in img_files:
                continue
            if prefix not in ann_files:
                continue
            img_infos.append(dict(
                img_file=img_files[prefix],
                ann_file=ann_files[prefix],
                thing_category_ids=thing_category_ids,
                **info))
        return img_infos


@DATASETS.register_module()
class COCOPanopticValDataset(COCOPanopticDataset):
    """Dataset for the validation set of COCO panoptic segmentation task."""

    def __init__(self,
                 pipeline,
                 data_root,
                 ignore_prefix_file=None,
                 **kwargs):
        """
        Args:
            pipeline (list[dict]): A sequence of data transforms.
            data_root (str): Root directory where the dataset is stored.
            ignore_prefix_file (str, optional): Path to a file containing a list
                of image prefixes to ignore. Default to None.
            **kwargs: Additional keyword arguments.
        """
        super(COCOPanopticValDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            train=False,
            ignore_prefix_file=ignore_prefix_file,
            **kwargs
        )

        # Get image info from dataset
        img_infos = []
        for sample in self.img_infos:
            segments_info = sample.pop('segments_info')
            for info in segments_info:
                img_info = deepcopy(sample)
                img_info['segments_info'] = [info]
                img_infos.append(img_info)

        self.img_infos = img_infos
