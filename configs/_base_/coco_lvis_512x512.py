crop_size = (512, 512)

pipeline = [
    dict(type='Resize', img_scale=None, ratio_range=(0.5, 2.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ObjectSampler',
        num_samples=1,
        max_num_merged_objects=1,
        min_area_ratio=0.0,
        merge_prob=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=('img', 'gt_semantic_seg'))]

coco_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='EncodeCOCOPanopticAnnotations')] + pipeline

lvis_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TransformLVISAnnotations')] + pipeline

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=[
        dict(
            type='COCOPanopticDataset',
            data_root='data/coco2017',
            pipeline=coco_pipeline,
            ignore_prefix_file='data/interformer/ignore_prefix.json'),
        dict(
            type='LVISDataset',
            data_root='data/lvis',
            pipeline=lvis_pipeline,
            ignore_prefix_file='data/interformer/ignore_prefix.json')])
