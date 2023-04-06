custom_imports = dict(imports=['interformer'], allow_failed_imports=False)

# model
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='ClickSegmentor',
    pretrained='pretrain/mae_pretrain_vit_large_mmcls.pth',
    backbone=dict(
        type='MAEWithSimpleFPN',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=-1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=1.0,
        fpn_cfg=dict(
            type='SimpleFPN',
            rescales=(4, 2, 1, 0.5),
            out_dims=(32, 64, 128, 256))),
    neck=dict(
        type='IMSA',
        target_size=(512, 512),
        in_channels=[32, 64, 128, 256],
        num_heads=(1, 2, 4, 8),
        depths=(2, 2, 6, 3),
        refine_start=(1, 1, 2, 1),
        mlp_ratio=(8, 8, 4, 4)),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=64,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='NormalizedFocalLoss', loss_weight=1.0),
            dict(type='BinaryIoU')]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='NormalizedFocalLoss', loss_weight=0.4)),
    train_cfg=dict(
        max_num_clicks=20,
        gamma=0.6,
        inner_radius=5,
        outer_radius=0,
        sfc_inner_k=1.7),
    test_cfg=dict(
        num_clicks=20,
        inner_radius=5,
        outer_radius=0))


# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.75,
        custom_keys={
            'patch_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}))
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
