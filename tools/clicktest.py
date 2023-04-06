import argparse
import os
import os.path as osp
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import mmcv
from mmcv.utils.misc import to_2tuple
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.engine import collect_results_cpu, collect_results_gpu

from mmseg import digit_version
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Command line arguments
    """
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir',
                        help='if specified, the evaluation metric results will be dumped into the directory as json')
    parser.add_argument('--gpu-collect', action='store_true',
                        help='whether to use gpu to collect results.')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of gpu to use (only applicable to non-distributed testing)')
    parser.add_argument('--tmpdir',
                        help='tmp directory used for collecting results from multiple workers, '
                             'available when gpu_collect is not specified')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num_clicks', default=20, type=int)
    parser.add_argument('--inner_radius', default=5, type=float)
    parser.add_argument('--outer_radius', default=0, type=float)
    parser.add_argument('--noc_target', default=[0.85, 0.9], nargs='+', type=float)
    parser.add_argument('--size', default=None, type=int)
    parser.add_argument('--size_divisor', default=32, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def set_test_cfg(cfg, args):
    """
    Set test configuration for the model.

    :param cfg: Config object.
    :param args: Arguments passed to the function.
    """
    pipeline = [
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
            to_rgb=True),
        dict(
            type='Pad',
            size=(None if args.size is None else to_2tuple(args.size)),
            size_divisor=(args.size_divisor if args.size is None else None),
            pad_val=0,
            seg_pad_val=0),
        dict(
            type='ObjectSampler',
            num_samples=1,
            max_num_merged_objects=1,
            min_area_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=('img', 'gt_semantic_seg'))
    ]

    dataset = args.dataset.lower()
    if dataset.startswith('sbd'):
        if 'sbddataset'.startswith(dataset):
            dataset = 'SBDDataset'
        elif 'sbdsubdataset'.startswith(dataset):
            dataset = 'SBDSubDataset'  # 10% samples of SBD for fast validation
        else:
            raise NotImplementedError(f'cannot handle dataset {args.dataset}')
        data_test_cfg = dict(
            type=dataset,
            data_root='data/sbd/benchmark_RELEASE/dataset',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadSBDAnnotations')] + pipeline)
    elif dataset == 'davis':
        dataset = 'DAVISDataset'
        data_test_cfg = dict(
            type=dataset,
            data_root='data/davis',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations')] + pipeline)
    elif dataset == 'grabcut':
        dataset = 'GrabCutDataset'
        data_test_cfg = dict(
            type=dataset,
            data_root='data/grabcut',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadGrabCutAnnotations')] + pipeline)
    elif dataset == 'berkeley':
        dataset = 'BerkeleyDataset'
        data_test_cfg = dict(
            type=dataset,
            data_root='data/berkeley',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadBerkeleyAnnotations')] + pipeline)
    else:
        raise NotImplementedError(f'Cannot handle dataset {args.dataset}')

    model_test_cfg = dict(
        num_clicks=args.num_clicks,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius)
    cfg.merge_from_dict({
        'model.test_cfg': model_test_cfg,
        'data.test': data_test_cfg})

    return cfg


def single_gpu_intertest(model, data_loader):
    model.eval()
    results = list()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        results.append(model(**data, interact=True))
        prog_bar.update()
    return results


def multi_gpu_intertest(model, data_loader, tmpdir, gpu_collect=False):
    model.eval()
    results = list()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        results.append(model(**data, interact=True))
        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def main():
    args = parse_args()
    config = str(next(Path(args.checkpoint).parent.glob('*.py')))
    cfg = mmcv.Config.fromfile(config)

    # set test config to perform click interaction
    cfg = set_test_cfg(cfg, args)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    json_file = 'clicktest_{dataset}_{checkpoint}_{timestamp}.json'.format(
        dataset=args.dataset.lower(),
        checkpoint=Path(args.checkpoint).stem,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    if args.work_dir is None and rank == 0:
        work_dir = str(Path(args.checkpoint).parent)
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, json_file)
    elif rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, json_file)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_intertest(model, data_loader)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        results = multi_gpu_intertest(
            model,
            data_loader,
            tmpdir=args.tmpdir,
            gpu_collect=args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        print(f'\ncheckpoint at {args.checkpoint}: ')
        metric_dict = dict(
            checkpoint=args.checkpoint,
            iou_per_click=results)
        if args.size is None:
            metric_dict.update(dict(size_divisor=args.size_divisor))
        else:
            metric_dict.update(dict(size=args.size))
        metric_dict.update(dict(
            num_clicks=args.num_clicks,
            inner_radius=args.inner_radius,
            outer_radius=args.outer_radius))
        for target in args.noc_target:
            metric = f'NoC{int(target * 100):d}'
            results = np.array(metric_dict['iou_per_click'])
            success = (results.max(-1) >= target).mean()
            success = float(success)
            results = np.concatenate(
                [results[:, :-1], np.ones((len(results), 1))], axis=-1)
            noc = (results >= target).argmax(-1).mean() + 1
            noc = float(noc)
            print(f'{metric}: {noc:.2f} Success ratio: {success:.2f}')
            metric_dict.update({metric: (noc, success)})
        mmcv.dump(metric_dict, json_file, indent=4)


if __name__ == '__main__':
    main()
