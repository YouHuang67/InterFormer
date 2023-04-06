from pathlib import Path
import argparse

import mmcv
from mmseg.apis import init_segmentor
import tkinter as tk
from demo.gui.app import InteractiveDemoApp


def main():
    args = parse_args()

    # get the config and checkpoint
    config = str(next(Path(args.checkpoint).parent.glob('*.py')))
    cfg = mmcv.Config.fromfile(config)
    cfg = set_test_cfg(cfg, args)
    cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

    # build the model
    model = init_segmentor(cfg, args.checkpoint, device=args.device)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--size_divisor', default=32, type=int)
    args = parser.parse_args()
    return args


def set_test_cfg(cfg, args):
    data_test_cfg = dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                img_scale=None,
                ratio_range=(1.0, 1.0),
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=False),
            dict(
                type='Pad',
                size=None,
                size_divisor=args.size_divisor,
                pad_val=0,
                seg_pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])])
    cfg.merge_from_dict({
        'model.backbone.type': 'MAEWithSimpleFPN',
        'model.type': 'ClickSegmentorZoomIn',
        'data.test': data_test_cfg})
    return cfg


if __name__ == '__main__':
    main()
