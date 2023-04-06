from copy import deepcopy
from pathlib import Path
from mmseg.datasets import CustomDataset
from mmseg.datasets.pipelines import Compose

from .. import utils


class BaseDataset(CustomDataset):
    """
    Base dataset class for semantic segmentation tasks.
    """

    CLASSES = ('Background', 'Foreground')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(
        self,
        pipeline,
        data_root,
        img_suffix='.jpg',
        ann_suffix='.png',
        ignore_prefix=None,
        ignore_index=255,
        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
    ):
        """
        Args:
            pipeline (list[dict]): list of pipeline configurations.
            data_root (str): dataset root directory.
            img_suffix (str): image filename suffix. Default: '.jpg'.
            ann_suffix (str): annotation filename suffix. Default: '.png'.
            ignore_prefix (str): prefix to ignore in filenames. Default: None.
            ignore_index (int): index to ignore in annotation masks. Default: 255.
            reduce_zero_label (bool): whether to reduce the label values by 1.
                Default: False.
            gt_seg_map_loader_cfg (list[dict]): list of pipeline configurations
                for loading ground truth segmentation maps. Default: None.
        """
        self.pipeline = Compose(pipeline)
        self.data_root = Path(data_root)
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        if ignore_prefix is None:
            self.ignore_prefix = set()
        else:
            self.ignore_prefix = set(ignore_prefix)
        self.img_infos = self.load_annotations()

        # Initialize the following variables to be compatible with the evaluation
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        if gt_seg_map_loader_cfg is None:
            self.gt_seg_map_loader = None
        else:
            self.gt_seg_map_loader = Compose(gt_seg_map_loader_cfg)

    def load_annotations(self):
        """
        Abstract method to be implemented by subclasses. Loads dataset
        annotations and returns a list of image information dictionaries.
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.img_infos)

    def __getitem__(self, item):
        """
        Retrieves the specified item from the dataset and returns it after
        applying the pipeline transformations.

        Args:
            item (int): item index.

        Returns:
            dict: dictionary containing the transformed image and annotations.
        """
        for _ in range(utils.MAX_NUM_LOOPS):
            try:
                return self.pipeline(self.pre_pipeline(item))
            except (utils.InvalidAnnotationError, utils.InfiniteLoopError):
                pass
        else:
            prefix = self.img_infos[item]['img_file'].stem
            raise utils.InvalidSample(f'invalid sample with prefix: {prefix}')

    def pre_pipeline(self, index):
        """
        Performs pre-pipeline transformations on the specified item index.

        Args:
            index (int): item index.

        Returns:
            dict: dictionary containing pre-processed image information.
        """
        img_info = deepcopy(self.img_infos[index])
        img_file = img_info.pop('img_file')
        ann_file = img_info.pop('ann_file', None)
        results = dict(
            seg_fields=list(),
            img_prefix=str(img_file.parent),
            img_info=dict(filename=img_file.name),
            **img_info
        )
        if ann_file is not None:
            results.update(dict(
                seg_prefix=str(ann_file.parent),
                ann_info=dict(seg_map=ann_file.name)
            ))
        return results

    def get_gt_seg_map_by_idx(self, index):
        """
        Get the ground truth semantic segmentation map for an image.

        Args:
            index (int): index of the image.

        Returns:
            torch.Tensor: ground truth semantic segmentation tensor with shape
                (1, H, W).
        """
        assert self.gt_seg_map_loader is not None, 'gt_seg_map_loader is None'
        results = self.gt_seg_map_loader(self.pre_pipeline(index))
        return results['gt_semantic_seg']
