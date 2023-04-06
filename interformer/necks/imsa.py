from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

import mmcv
from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.runner import BaseModule, Sequential
from mmcv.cnn import build_norm_layer
from mmcv.utils.misc import to_2tuple
from mmcv.cnn.bricks.drop import build_dropout
from mmseg.models.builder import NECKS
from mmseg.models.utils.embed import PatchMerging

from ..utils import rearrange

IMSA_BLOCKS = Registry('IMSA_blocks', parent=MMCV_MODELS)
def build_imsa_block(cfg):
    return IMSA_BLOCKS.build(cfg)


class PoolingAttention(BaseModule):

    def __init__(self,
                 target_size,  # used to limit the minimum size of pooling features
                 embed_dims,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 pool_ratios=(1, 2, 3, 6),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        target_size = tuple(to_2tuple(target_size))
        assert len(target_size) == 2
        assert mmcv.is_tuple_of(tuple(pool_ratios), int)
        super(PoolingAttention, self).__init__(init_cfg)
        self.target_size = target_size
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.pool_ratios = pool_ratios

        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.query_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.key_value_proj = nn.Linear(
            embed_dims, 2 * embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)
        _, self.norm = build_norm_layer(norm_cfg, embed_dims)
        self.dconvs = nn.ModuleList([nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims) for _ in pool_ratios])

    def forward(self, x, kv_in):
        """
        :type x: torch.Tensor
        :type kv_in: torch.Tensor
        :rtype: torch.Tensor

        :param x: shape (batch_size, num_channels, down_height, down_width)
        :param kv_in: (batch_size, num_channels, height, width)
        :return: shape (batch_size, num_channels, down_height, down_width)
        """
        if x.dim() != 4:
            raise ValueError(f'`x` is expected to have the dim of 4, '
                             f'but got shape {tuple(x.size())} with '
                             f'dim of {x.dim()}')
        if kv_in.dim() != 4:
            raise ValueError(f'`kv_in` is expected to have the dim of 4, '
                             f'but got shape {tuple(kv_in.size())} with '
                             f'dim of {kv_in.dim()}')
        if x.size()[:2] != kv_in.size()[:2]:
            raise ValueError(f'`x` and `kv_in` are expected to have '
                             f'the same batch_size and num_channels, '
                             f'but got the shape {tuple(x.size())} and '
                             f'{tuple(kv_in.size())}')

        hd, wd = x.size()[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        query = self.query_proj(x)

        kvs = list()
        h, w = kv_in.size()[-2:]
        for ratio, dconv in zip(self.pool_ratios, self.dconvs):
            if ratio == 1:
                kv = kv_in
            else:
                hm, wm = self.target_size
                hp = min(hm / ratio, h)
                hp = round(h / round(h / hp))
                wp = min(wm / ratio, w)
                wp = round(w / round(w / wp))
                if hp == h and wp == w:
                    kv = kv_in
                else:
                    kv = F.adaptive_max_pool2d(kv_in, (hp, wp))
            kv = kv + dconv(kv)
            kvs.append(rearrange(kv, 'b c h w -> b (h w) c'))
        kv_in = self.norm(torch.cat(kvs, dim=1))
        key, value = torch.chunk(self.key_value_proj(kv_in), 2, dim=-1)

        query = rearrange(query, 'b n (hn c) -> b hn n c', hn=self.num_heads)
        key = rearrange(key, 'b n (hn c) -> b hn n c', hn=self.num_heads)
        value = rearrange(value, 'b n (hn c) -> b hn n c', hn=self.num_heads)

        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ value
        x = rearrange(x, 'b hn n c -> b n (hn c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hd, w=wd)
        return x


class InvertedBottleneck(Sequential):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3):
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.kernel_size = kernel_size
        super(InvertedBottleneck, self).__init__(
            nn.Conv2d(embed_dims, feedforward_channels, 1, 1, 0),
            nn.Hardswish(),
            nn.Conv2d(feedforward_channels,
                      feedforward_channels,
                      kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2,
                      stride=1,
                      groups=feedforward_channels),
            nn.Hardswish(),
            nn.Conv2d(feedforward_channels, embed_dims, 1, 1, 0))


@IMSA_BLOCKS.register_module()
class PoolingBlock(BaseModule):

    def __init__(self,
                 target_size,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pool_ratios=(12, 16, 20, 24),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(PoolingBlock, self).__init__(init_cfg)
        self.norm1 = nn.GroupNorm(1, embed_dims)
        self.norm_kv = nn.GroupNorm(1, embed_dims)
        self.attn = PoolingAttention(
            target_size=target_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            pool_ratios=pool_ratios,
            norm_cfg=norm_cfg,
            init_cfg=None)

        self.norm2 = nn.GroupNorm(1, embed_dims)
        self.mlp = InvertedBottleneck(
            embed_dims=embed_dims,
            feedforward_channels=int(mlp_ratio * embed_dims))
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, x, kv_in=None):
        """
        :type x: torch.Tensor
        :type kv_in: Optional[torch.Tensor]
        :rtype: torch.Tensor

        :param x: shape (batch_size, num_channels, down_height, down_width)
        :param kv_in: shape (batch_size, num_channels, height, width)
        :return: shape (batch_size, num_channels, down_height, down_width)
        """
        kv_in = x if kv_in is None else kv_in
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm_kv(kv_in)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@IMSA_BLOCKS.register_module()
class PoolingBlockSequence(BaseModule):

    def __init__(self,
                 target_size,
                 embed_dims,
                 num_heads,
                 depth,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 pool_ratios=(12, 16, 20, 24),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 block_cfg=dict(type='PoolingBlock')):
        super(PoolingBlockSequence, self).__init__(init_cfg)
        if isinstance(drop_path_rate, (list, tuple)):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            cfg = deepcopy(block_cfg)
            cfg.update(dict(
                target_size=target_size,
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                pool_ratios=pool_ratios,
                norm_cfg=norm_cfg,
                init_cfg=None))
            block = build_imsa_block(cfg)
            self.blocks.append(block)
        self.downsample = downsample

    def forward(self, x, kv_in):
        """
        :type x: torch.Tensor
        :type kv_in: torch.Tensor
        :rtype: Tuple[torch.Tensor, torch.Tensor]

        :param x: shape (batch_size, num_channels, down_height, down_width)
        :param kv_in: shape (batch_size, num_channels, height, width)
        """
        for i, block in enumerate(self.blocks):
            extra = dict() if i > 0 else dict(kv_in=kv_in)
            x = block(x, **extra)
        if self.downsample is None:
            return x, x
        else:
            h, w = x.size()[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
            x_down, (hd, wd) = self.downsample(x, (h, w))
            x_down = rearrange(x_down, 'b (h w) c -> b c h w', h=hd, w=wd)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x_down, x


@IMSA_BLOCKS.register_module()
class RefineBlockSequence(PoolingBlockSequence):

    def __init__(self,
                 target_size,
                 embed_dims,
                 num_heads,
                 depth,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pool_ratios=(12, 16, 20, 24),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 block_cfg=dict(type='PoolingBlock')):
        super(RefineBlockSequence, self).__init__(
            target_size=target_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            downsample=None,
            pool_ratios=pool_ratios,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            block_cfg=block_cfg)

    @staticmethod
    def merge_roi(roi, hw_shape):
        """
        :type roi: torch.Tensor
        :type hw_shape: Tuple[int, int]
        :rtype: torch.Tensor

        :param roi: shape (batch_size, 4)
        :param hw_shape: (height, width)
        :return: shape (batch_size, 4)
        """
        left, up, right, bottom = torch.chunk(roi.long(), 4, dim=-1)
        heights, widths = bottom - up, right - left
        max_height, max_width = heights.max().item(), widths.max().item()
        left = torch.ceil((left + right) / 2.0 - max_width / 2.0)
        left = torch.clip(left, 0)
        right = left + max_width
        right = torch.clip(right, None, hw_shape[1])
        left = right - max_width
        up = torch.ceil((up + bottom) / 2.0 - max_height / 2.0)
        up = torch.clip(up, 0)
        bottom = up + max_height
        bottom = torch.clip(bottom, None, hw_shape[0])
        up = bottom - max_height
        return torch.cat([left, up, right, bottom], dim=-1).long()

    @staticmethod
    def roi_to_mask(roi, hw_shape):
        """
        :type roi: torch.Tensor
        :type hw_shape: Tuple[int, int]
        :rtype: torch.Tensor

        :param roi: shape (batch_size, 4)
        :param hw_shape: (height, width)
        :return: shape (batch_size, height, width)
        """
        h, w = hw_shape
        left, up, right, bottom = torch.chunk(roi.long(), 4, dim=-1)
        lower = torch.cat([up, left], dim=-1)
        lower = lower.view(roi.size(0), 1, 1, 2).repeat(1, h, w, 1)
        upper = torch.cat([bottom, right], dim=-1)
        upper = upper.view(roi.size(0), 1, 1, 2).repeat(1, h, w, 1)
        idxs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        idxs = torch.stack(idxs, dim=-1).to(roi)
        idxs = idxs.view(1, h, w, 2).repeat(roi.size(0), 1, 1, 1)
        mask = torch.logical_and(
            torch.greater_equal(idxs, lower), torch.less(idxs, upper))
        mask = mask.long().min(-1)[0].to(roi)
        return mask

    @staticmethod
    def check_input(x, roi):
        if x.dim() != 4:
            raise ValueError(f'`x` is expected to have the dim of 4, '
                             f'but got shape {tuple(x.size())} with '
                             f'dim of {x.dim()}')
        if roi is None:
            return x, roi
        if x.size(0) != roi.size(0):
            raise ValueError(f'`x` and `roi` are expected to have '
                             f'the same batch_size, but got '
                             f'{x.size(0)} and {roi.size(0)}')
        if roi.dim() != 2 or roi.size(-1) != 4:
            raise ValueError(f'`roi` is expected to have the shape '
                             f'(batch_size, 4) with each sample including '
                             f'a bbox represented by (left, up, right, bottom), '
                             f'but got the shape: {tuple(roi.size())}')
        if (torch.round(roi.float()) - roi).abs().max().item() > 0.0:
            raise ValueError(f'each bbox point is expected to be integer, '
                             f'but got {roi.detach().cpu().numpy().tolist()}')
        return x, roi

    def forward(self, x, roi=None):
        """
        :type x: torch.Tensor
        :type roi: Optional[torch.Tensor]
        :rtype: torch.Tensor

        :param x: shape (batch_size, num_channels, height, width)
        :param roi: shape (batch_size, 4), each for left, up, right, bottom
        :return: shape (batch_size, num_channels, height, width)
        """
        x, roi = self.check_input(x, roi)
        if roi is None:
            for block in self.blocks:
                x = block(x, kv_in=x)
            return x

        if x.size(0) == 1:
            left, up, right, bottom = \
                roi.view(4).long().detach().cpu().numpy().tolist()
            crop_x = x[..., up:bottom, left:right]
            for block in self.blocks:
                crop_x = block(crop_x, kv_in=crop_x)
            if self.training:
                ori_x = x
                x = torch.empty_like(x)
                x[..., up:bottom, left:right] = crop_x
                mask = self.roi_to_mask(roi, x.size()[-2:]).float()
                x = (1 - mask) * ori_x + mask * x
            else:
                x = x.clone()
                x[..., up:bottom, left:right] = crop_x

        else:
            ext_roi = self.merge_roi(roi, x.size()[-2:])
            left, up, right, bottom = torch.chunk(ext_roi.long(), 4, dim=-1)
            shape = (torch.unique(bottom - up).item(),
                     torch.unique(right - left).item())
            rel_roi = ext_roi - roi[..., :2].repeat(1, 2)
            idx = torch.arange(roi.size(0)).view(-1, 1).to(roi)
            ext_roi = torch.cat([idx, ext_roi], dim=-1).to(x)
            rel_roi = torch.cat([idx, rel_roi], dim=-1).to(x)
            crop_x = roi_align(x, ext_roi, shape, aligned=True)
            for block in self.blocks:
                kv_in = roi_align(crop_x, rel_roi, shape, aligned=True)
                crop_x = block(crop_x, kv_in=kv_in)
            ori_x = x
            x = torch.empty_like(x)
            x = rearrange(x, 'b c h w -> b h w c')
            mask = self.roi_to_mask(roi, x.size()[-2:]).bool()
            crop_mask = roi_align(
                mask.unsqueeze(1).float(),
                ext_roi,
                shape,
                aligned=True).squeeze(1).bool()
            x[~mask] = rearrange(ori_x, 'b c h w -> b h w c')[~mask]
            x[mask] = rearrange(crop_x, 'b c h w -> b h w c')[crop_mask]
            x = rearrange(x, 'b h w c -> b c h w')
        return x


@IMSA_BLOCKS.register_module()
class GlobalPoolingRefiner(RefineBlockSequence):

    def forward(self, x, roi=None):
        """
        :type x: torch.Tensor
        :type roi: Optional[torch.Tensor]
        :rtype: torch.Tensor

        :param x: shape (batch_size, num_channels, height, width)
        :param roi: shape (batch_size, 4), each for left, up, right, bottom
        :return: shape (batch_size, num_channels, height, width)
        """
        x, roi = self.check_input(x, roi)
        if roi is None:
            for block in self.blocks:
                x = block(x, kv_in=x)
            return x

        if x.size(0) == 1:
            left, up, right, bottom = \
                roi.view(4).long().detach().cpu().numpy().tolist()
            mask = self.roi_to_mask(roi, x.size()[-2:]).float()
            crop_x = x[..., up:bottom, left:right]
            for block in self.blocks:
                crop_x = block(crop_x, kv_in=x)
                if self.training:
                    ori_x = x
                    x = torch.empty_like(x)
                    x[..., up:bottom, left:right] = crop_x
                    x = (1 - mask) * ori_x + mask * x
                else:
                    x = x.clone()
                    x[..., up:bottom, left:right] = crop_x

        else:
            ext_roi = self.merge_roi(roi, x.size()[-2:])
            left, up, right, bottom = torch.chunk(ext_roi.long(), 4, dim=-1)
            shape = (torch.unique(bottom - up).item(),
                     torch.unique(right - left).item())
            idx = torch.arange(roi.size(0)).view(-1, 1).to(roi)
            ext_roi = torch.cat([idx, ext_roi], dim=-1).to(x)
            mask = self.roi_to_mask(roi, x.size()[-2:]).bool()
            crop_mask = roi_align(
                mask.unsqueeze(1).float(),
                ext_roi,
                shape,
                aligned=True).squeeze(1).bool()
            crop_x = roi_align(x, ext_roi, shape, aligned=True)
            for block in self.blocks:
                crop_x = block(crop_x, kv_in=x)
                ori_x = x
                x = torch.empty_like(x)
                x = rearrange(x, 'b c h w -> b h w c')
                x[~mask] = rearrange(ori_x, 'b c h w -> b h w c')[~mask]
                x[mask] = rearrange(crop_x, 'b c h w -> b h w c')[crop_mask]
                x = rearrange(x, 'b h w c -> b c h w')
        return x


@NECKS.register_module()
class IMSA(BaseModule):

    def __init__(self,
                 target_size,
                 in_channels,
                 num_heads,
                 depths,
                 refine_start,
                 mlp_ratio=4.,
                 strides=(4, 2, 2, 2),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 patch_kernel_size=3,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 pool_ratios=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 num_ref_modes=5,
                 coarse_cfg=dict(type='PoolingBlockSequence'),
                 refine_cfg=dict(type='RefineBlockSequence')):
        super(IMSA, self).__init__(init_cfg=init_cfg)
        target_size = tuple(to_2tuple(target_size))
        if len(target_size) != 2:
            raise ValueError(f'Invalid `target_size` that is expected to be '
                             f'an integer or a tuple (height, width), '
                             f'but got {target_size}')
        if not (mmcv.is_tuple_of(refine_start, int) or
                mmcv.is_list_of(refine_start, int)):
            raise ValueError(f'Invalid values of `refine_start` that '
                             f'is expected to be a list/tuple of int, '
                             f'but got {refine_start}')
        num_stages = len(depths)
        self.num_stages = num_stages
        if len(refine_start) != num_stages:
            raise ValueError(f'`refine_start` and `depths` are expected to '
                             f'have the same length, but got '
                             f'{len(refine_start)} and {num_stages}')
        if not mmcv.is_list_of(in_channels, int):
            raise ValueError(f'Invalid values of `in_channels` that '
                             f'is expected to be a list of int, '
                             f'but got {in_channels}')
        if len(in_channels) != num_stages:
            raise ValueError(f'`in_channels` and `depths` are expected to '
                             f'have the same length, but got '
                             f'{len(in_channels)} and {num_stages}')
        if len(num_heads) != num_stages:
            raise ValueError(f'`num_heads` and `depths` are expected to '
                             f'have the same length, but got '
                             f'{len(num_heads)} and {num_stages}')
        if len(strides) != num_stages:
            raise ValueError(f'`strides` and `depths` are expected to '
                             f'have the same length, but got '
                             f'{len(strides)} and {num_stages}')
        self.in_channels = in_channels
        self.strides = strides
        if isinstance(mlp_ratio, (list, tuple)):
            mlp_ratios = mlp_ratio
            if len(mlp_ratio) != num_stages:
                raise ValueError(f'`mlp_ratio` and `depths` are expected to '
                                 f'have the same length, but got '
                                 f'{len(mlp_ratio)} and {num_stages}')
        else:
            mlp_ratios = [deepcopy(mlp_ratio) for _ in depths]
        if pool_ratios is None:
            pool_ratios = [
                (12, 16, 20, 24),
                (6, 8, 10, 12),
                (3, 4, 5, 6),
                (1, 2, 3, 4)]
        if not mmcv.is_list_of(pool_ratios, (list, tuple)):
            raise ValueError(f'`pool_ratios` is expected to be a list of '
                             f'list/tuple, but got {pool_ratios}')

        def down_size(size, stride):
            return size[0] // stride, size[1] // stride

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages, self.refine_stages = nn.ModuleList(), nn.ModuleList()
        target_size, strides = down_size(target_size, strides[0]), strides[1:]
        for i in range(num_stages):
            if i < num_stages - 1:
                downsample = PatchMerging(
                    in_channels=in_channels[i],
                    out_channels=in_channels[i + 1],
                    kernel_size=patch_kernel_size,
                    stride=strides[i],
                    padding=(patch_kernel_size - 1) // 2,
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            dp = dpr[sum(depths[:i]):sum(depths[:i + 1])]
            cfg = deepcopy(coarse_cfg)
            cfg.update(dict(
                target_size=target_size,
                embed_dims=in_channels[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                depth=refine_start[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dp[:refine_start[i]],
                downsample=downsample,
                pool_ratios=pool_ratios[i],
                norm_cfg=norm_cfg,
                init_cfg=None))
            stage = build_imsa_block(cfg)
            self.stages.append(stage)
            self.add_module(f'norm{i}', nn.GroupNorm(1, self.in_channels[i]))

            if depths[i] - refine_start[i] > 0:
                cfg = deepcopy(refine_cfg)
                cfg.update(dict(
                    target_size=target_size,
                    embed_dims=in_channels[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    depth=(depths[i] - refine_start[i]),
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dp[refine_start[i]:],
                    pool_ratios=pool_ratios[i],
                    norm_cfg=norm_cfg,
                    init_cfg=None))
                stage = build_imsa_block(cfg)
            else:
                stage = nn.Identity()
            self.refine_stages.append(stage)
            if i < num_stages - 1:
                target_size = down_size(target_size, strides[i])

        self.num_ref_modes = num_ref_modes
        patch_embed = torch.empty(num_ref_modes, in_channels[0])
        patch_embed = nn.init.xavier_normal_(patch_embed)
        self.patch_embed = nn.Parameter(patch_embed)

    def ref_embed(self, x, ref_label):
        """
        :type x: torch.Tensor
        :type ref_label: torch.Tensor
        :rtype: torch.Tensor

        :param x: shape (batch_size, num_channels, down_height, down_width)
        :param ref_label: shape (batch_size, 1, height, width)
        :return: shape (batch_size, num_channels, down_height, down_width)
        """
        ref_mask = F.one_hot(ref_label.long(), self.num_ref_modes)
        patch_embed = ref_mask.float() @ self.patch_embed
        patch_embed = rearrange(patch_embed, 'b () h w c -> b c h w')
        if patch_embed.size()[-2:] != x.size()[-2:]:
            patch_embed = F.interpolate(
                patch_embed, x.size()[-2:], mode='bilinear')
        return x + patch_embed

    def stem(self, x, inputs, roi):
        """
        :param x: torch.Tensor,
            shape: (batch_size, num_channels, height, width)
        :param inputs: a list of torch.Tensor with similar shape of x
        :param roi: torch.Tensor, shape: (batch_size, 4)
        :return: a list of torch.Tensor sharing the same shapes of inputs
        """
        outs = []
        for i, stage in enumerate(self.stages):
            x, out = stage(x, kv_in=inputs[i])
            roi = None if roi is None else (roi / self.strides[i])
            if not isinstance(self.refine_stages[i], nn.Identity):
                out = self.refine_stages[i](out, roi)
            norm_layer = getattr(self, f'norm{i}')
            outs.append(norm_layer(out))
        return outs

    def forward(self, inputs, ref_label, roi=None):
        if not (mmcv.is_list_of(inputs, torch.Tensor) or
                mmcv.is_tuple_of(inputs, torch.Tensor)):
            raise TypeError(f'`inputs` is expected to be a list/tuple of '
                            f'torch.Tensor, but got a '
                            f'{str(type(inputs)).lower()} of '
                            f'{type(inputs[0])}')
        if len(inputs) != self.num_stages:
            raise ValueError(f'`inputs` is expected to have {self.num_stages} '
                             f'torch.Tensor corresponding to the stages, '
                             f'but got {len(inputs)} torch.Tensor')
        if roi is not None and ref_label.dim() != 4:
            raise ValueError(f'`ref_label` is expected to have a dim of 4, '
                             f'but got shape {tuple(ref_label.size())} '
                             f'with a dim of {ref_label.dim()}')
        if roi is not None and (roi.dim() != 3 or roi.size(-1) != 4):
            raise ValueError(f'`roi` is expected to have a shape of (*, *, 4) '
                             f'with a dim of 3, but got shape '
                             f'{tuple(roi.size())}')
        if roi is not None and ref_label.size()[:2] != roi.size()[:2]:
            raise ValueError(f'`ref_label` and `roi` are expected to '
                             f'have the same batch_size and num_samples, '
                             f'but got their shapes {ref_label.size()} '
                             f'and {roi.size()}')
        if ref_label is not None and ref_label.size(1) > 1:
            num_instances = ref_label.size(1)
            inputs = [x.repeat(num_instances, 1, 1, 1) for x in inputs]
            ref_label = rearrange(ref_label, 'b n h w -> (b n) () h w')
            if roi is not None:
                roi = rearrange(roi, 'b n c4 -> (b n) () c4')
        x = self.ref_embed(inputs[0], ref_label)
        roi = None if roi is None else roi.squeeze(1)
        inputs = self.stem(x, inputs, roi)
        return inputs
