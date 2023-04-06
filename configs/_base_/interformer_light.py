_base_ = ['interformer_tiny.py']
model = dict(
    pretrained='pretrain/mae_pretrain_vit_base_mmcls.pth',
    backbone=dict(embed_dims=768, num_layers=12, num_heads=12),
    neck=dict(depths=(1, 1, 2, 1)))
optimizer = dict(paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))
