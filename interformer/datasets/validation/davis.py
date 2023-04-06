from mmseg.datasets.builder import DATASETS
from ..base import BaseDataset


@DATASETS.register_module()
class DAVISDataset(BaseDataset):

    def __init__(self, pipeline, data_root, **kwargs):
        super(DAVISDataset, self).__init__(
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
