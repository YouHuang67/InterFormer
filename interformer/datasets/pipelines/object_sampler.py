import random
import numpy as np
from numpy import random as rng
from mmseg.datasets.builder import PIPELINES
from .. import utils


@PIPELINES.register_module()
class ObjectSampler:
    """
    Sample objects from semantic segmentation annotation for data augmentation.
    """
    def __init__(
        self, num_samples=1, max_num_merged_objects=1, min_area_ratio=0.0,
        ignore_index=255, merge_prob=0.0, include_other=False
    ):
        """
        Args:
            num_samples (int): Number of times to sample objects.
            max_num_merged_objects (int): Maximum number of objects to merge.
            min_area_ratio (float): Minimum area ratio of objects to be selected.
            ignore_index (int): The index to be ignored.
            merge_prob (float): The probability of merging objects.
            include_other (bool): Whether to include other objects.
                These objects are not selected and are represented by 2 in the mask.
        """
        self.num_samples = num_samples
        self.max_num_merged_objects = max_num_merged_objects
        self.min_area_ratio = min_area_ratio
        self.ignore_index = ignore_index
        self.merge_prob = merge_prob
        self.include_other = include_other

    def __call__(self, results):
        gt_semantic_seg = results.pop('gt_semantic_seg')
        segments_info = []
        for info in results.pop('segments_info'):
            if np.any(gt_semantic_seg == info['id']):
                segments_info.append(info)
        if len(segments_info) == 0:
            raise utils.InvalidAnnotationError('Not found valid annotation')

        gt_semantic_segs = []
        for _ in range(self.num_samples):
            seg_label = self.sample_objects(gt_semantic_seg, segments_info)
            gt_semantic_segs.append(seg_label)

        if len(gt_semantic_segs) == 1:
            results['gt_semantic_seg'] = gt_semantic_segs[0]
        else:
            results['gt_semantic_seg'] = np.stack(gt_semantic_segs, 0)

        return results

    def sample_objects(self, gt_semantic_seg, segments_info):
        merge_prob = self.merge_prob
        ignore_index = self.ignore_index
        max_num_merged_objects = self.max_num_merged_objects
        num_objects = len(segments_info)

        for _ in range(utils.MAX_NUM_LOOPS):
            seg_label = np.zeros_like(gt_semantic_seg)
            object_idxs = rng.permutation(range(num_objects))

            if max_num_merged_objects > 1 and random.random() < merge_prob:
                num_merged_objects = random.randint(2, max_num_merged_objects)
            else:
                num_merged_objects = 1

            for idx in object_idxs[:num_merged_objects]:
                seg_label[gt_semantic_seg == segments_info[idx]['id']] = 1

            if np.mean(seg_label) > self.min_area_ratio:
                seg_label[gt_semantic_seg == ignore_index] = ignore_index

                if not self.include_other:
                    return seg_label

                for idx in object_idxs[num_merged_objects:]:
                    seg_label[gt_semantic_seg == segments_info[idx]['id']] = 2

                return seg_label

        else:
            raise utils.InfiniteLoopError(
                f'Not found valid objects with area ratio '
                f'larger than {self.min_area_ratio}'
            )

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_samples={self.num_samples}, '
                f'max_num_merged_objects={self.max_num_merged_objects}, '
                f'min_area_ratio={self.min_area_ratio}, '
                f'ignore_index={self.ignore_index}, '
                f'merge_prob={self.merge_prob}, '
                f'include_other={self.include_other})')
