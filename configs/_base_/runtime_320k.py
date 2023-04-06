_base_ = ['runtime_40k.py']
runner = dict(type='IterBasedRunner', max_iters=320000)
checkpoint_config = dict(by_epoch=False, interval=20000)
