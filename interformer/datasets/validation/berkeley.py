import numpy as np
from PIL import Image
from pathlib import Path

from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.builder import DATASETS
from ..base import BaseDataset


@PIPELINES.register_module()
class LoadBerkeleyAnnotations(object):

    def __call__(self, results):
        ann_file = Path(results.pop('seg_prefix')) / \
                   results.pop('ann_info')['seg_map']
        gt_semantic_seg = np.array(Image.open(str(ann_file), 'r'))
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@DATASETS.register_module()
class BerkeleyDataset(BaseDataset):

    def __init__(self, pipeline, data_root, **kwargs):
        super(BerkeleyDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            img_suffix='.jpg',
            ann_suffix='.png',
            ignore_prefix=set(),
            ignore_index=255,
            reduce_zero_label=False,
            gt_seg_map_loader_cfg=None)

    def load_annotations(self):
        data_root = self.data_root
        img_suffix = self.img_suffix
        ann_suffix = self.ann_suffix
        img_files = {p.stem: p for p in data_root.rglob('*' + img_suffix)}
        ann_files = {p.stem: p for p in data_root.rglob('*' + ann_suffix)}
        img_infos = list()
        for prefix in sorted(img_files):
            img_infos.append(dict(
                img_file=img_files[prefix],
                ann_file=ann_files[f'{prefix}#001'],
                segments_info=[dict(id=1)]))
        return img_infos
