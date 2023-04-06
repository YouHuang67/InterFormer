from pathlib import Path
from scipy.io import loadmat

import mmcv
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.builder import DATASETS
from ..base import BaseDataset
from interformer import utils


@PIPELINES.register_module()
class LoadSBDAnnotations(object):

    def __call__(self, results):
        ann_file = Path(results.pop('seg_prefix')) / \
                   results.pop('ann_info')['seg_map']
        gt_semantic_seg = loadmat(str(ann_file))['GTinst'][0][0][0]
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@DATASETS.register_module()
class SBDDataset(BaseDataset):

    def __init__(self, pipeline, data_root, buggy_threshold=0.0, **kwargs):
        self.buggy_threshold = buggy_threshold
        super(SBDDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            img_suffix='.jpg',
            ann_suffix='.mat',
            ignore_prefix=set(),
            ignore_index=255,
            reduce_zero_label=False,
            gt_seg_map_loader_cfg=None)

    def load_annotations(self):
        data_root = self.data_root
        img_suffix = self.img_suffix
        ann_suffix = self.ann_suffix
        img_files = {p.stem: p for p in data_root.rglob('*' + img_suffix)}
        ann_files = {p.stem: p for p in data_root.rglob('*inst/*' + ann_suffix)}
        val_file = next(data_root.rglob('val.txt'))
        with open(val_file, 'r') as file:
            str_buggy_threshold = str(self.buggy_threshold).replace('.', 'p')
            split_file = val_file.parent / f'val_bt_{str_buggy_threshold}.json'
            split = mmcv.load(split_file) if split_file.is_file() else dict()
            img_infos = list()
            for prefix in file.readlines():
                prefix = prefix.strip()
                img_file, ann_file = img_files[prefix], ann_files[prefix]
                if prefix in split:
                    for idx in split[prefix]:
                        img_infos.append(dict(
                            img_file=img_file,
                            ann_file=ann_file,
                            segments_info=[dict(id=idx)]))
                    continue
                mask = loadmat(str(ann_file))['GTinst'][0][0][0]
                for idx, area in zip(*utils.get_label_ids_with_areas(mask)):
                    if self.buggy_threshold > 0.:
                        bbox = utils.get_bbox_from_mask(mask == idx)
                        bbox_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
                        if float(area) / bbox_area < self.buggy_threshold:
                            continue
                    img_infos.append(dict(
                        img_file=img_file,
                        ann_file=ann_file,
                        segments_info=[dict(id=idx)]))
                    split.setdefault(prefix, list()).append(idx)
            mmcv.dump(split, split_file)
        return img_infos


@DATASETS.register_module()
class SBDSubDataset(SBDDataset):

    NUM_SKIPPED_SAMPLES = 10

    def __init__(self, pipeline, data_root, buggy_threshold=0.0, **kwargs):
        super(SBDSubDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            buggy_threshold=buggy_threshold)
        img_infos = sorted(self.img_infos, key=self.sort_key)
        self.img_infos = img_infos[::self.NUM_SKIPPED_SAMPLES]

    @staticmethod
    def sort_key(info):
        prefix = info['img_file'].stem
        object_idx = info['segments_info'][0]['id']
        return f'{prefix}_{object_idx}'
