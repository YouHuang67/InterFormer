from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES
from mmseg.models.builder import NECKS, build_neck
from mmseg.models.backbones.mae import MAE, MAEAttention, MAETransformerEncoderLayer

from ..utils import rearrange


class FlexSizeMAEAttention(MAEAttention):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 bias='qv_bias',
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None,
                 rel_pos_embed_resize_mode='nearest',
                 **kwargs):
        super(FlexSizeMAEAttention, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            bias=bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=init_cfg,
            **kwargs)
        self.rel_pos_embed_resize_mode = rel_pos_embed_resize_mode

    @staticmethod
    def resize_rel_pos_embed_to_match_input(
            embed: torch.Tensor,
            window_size: Tuple[int, int],
            hw_shape: Tuple[int, int],
            mode: str
    ) -> torch.Tensor:
        """
        :param embed: Tensor of shape (batch_size, Wh * Ww + 1, Wh * Ww + 1)
        :param window_size: Tuple of (window_height, window_width)
        :param hw_shape: Tuple of (feature_height, feature_width)
        :param mode: Interpolation mode, either 'nearest' or 'bilinear'
        :return: Tensor of shape (batch_size, h * w + 1, h * w + 1)
        """
        Wh, Ww = window_size
        if embed.dim() != 3:
            raise ValueError(f'embed has invalid shape {tuple(embed.size())} '
                             f'with {embed.dim()} dimensions != 3')
        if embed.size(1) != (Wh * Ww + 1) or embed.size(2) != (Wh * Ww + 1):
            raise ValueError(f'embed`s shape {tuple(embed.size())} '
                             f'cannot match the window size {window_size}')
        up, bottom = torch.split(embed, [1, Wh * Ww], dim=1)
        up_left, up_right = torch.split(up, [1, Wh * Ww], dim=2)
        bottom_left, bottom_right = torch.split(bottom, [1, Wh * Ww], dim=2)

        def resize(x):
            return F.interpolate(x, hw_shape, mode=mode)

        shape = dict(Wh=Wh, Ww=Ww)
        up_right = rearrange(up_right, 'b () (Wh Ww) -> b () Wh Ww', **shape)
        up_right = resize(up_right)
        up_right = rearrange(up_right, 'b () h w -> b () (h w)')

        bottom_left = rearrange(
            bottom_left, 'b (Wh Ww) () -> b () Wh Ww', **shape)
        bottom_left = resize(bottom_left)
        bottom_left = rearrange(bottom_left, 'b () h w -> b (h w) ()')

        bottom_right = rearrange(
            bottom_right, 'b WhWw (Wh Ww) -> b WhWw Wh Ww', **shape)
        bottom_right = resize(bottom_right)
        bottom_right = rearrange(
            bottom_right, 'b (Wh Ww) h w -> b (h w) Wh Ww', **shape)
        bottom_right = resize(bottom_right)
        bottom_right = rearrange(bottom_right, 'b hw h w -> b (h w) hw')

        up = torch.cat([up_left, up_right], dim=2)
        bottom = torch.cat([bottom_left, bottom_right], dim=2)
        embed = torch.cat([up, bottom], dim=1)
        return embed

    def forward(self, x, hw_shape):
        B, N, C = x.shape

        if self.bias == 'qv_bias':
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            Wh = self.window_size[0]
            Ww = self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    Wh * Ww + 1, Wh * Ww + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if attn.size()[1:] != relative_position_bias.size():
                relative_position_bias = \
                    self.resize_rel_pos_embed_to_match_input(
                        embed=relative_position_bias,
                        window_size=self.window_size,
                        hw_shape=hw_shape,
                        mode=self.rel_pos_embed_resize_mode)
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlexSizeMAETransformerEncoderLayer(MAETransformerEncoderLayer):

    def build_attn(self, attn_cfg):
        self.attn = FlexSizeMAEAttention(**attn_cfg)

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.gamma_1 *
                               self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        return x


@NECKS.register_module()
class SimpleFPN(BaseModule):

    def __init__(self,
                 in_dim,
                 rescales=(4, 2, 1, 0.5),
                 out_dims=(64, 128, 256, 512)):
        assert len(rescales) == len(out_dims)
        super(SimpleFPN, self).__init__()
        self.in_dim = in_dim
        self.rescales = rescales
        self.out_dims = out_dims
        self.rescale_ops = nn.ModuleList()
        for k, out_dim in zip(self.rescales, self.out_dims):
            if k == 16:
                dim = max(in_dim // 2, 8 * out_dim)
                ops = nn.Sequential(
                    nn.ConvTranspose2d(in_dim, dim, 2, stride=2),
                    nn.GroupNorm(1, dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
                    nn.GroupNorm(1, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2),
                    nn.GroupNorm(1, dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 4, dim // 8, 2, stride=2),
                    nn.GroupNorm(1, dim // 8),
                    nn.Conv2d(dim // 8, out_dim, 1),
                    nn.GroupNorm(1, out_dim),
                    nn.GELU())
            elif k == 4:
                dim = max(in_dim // 2, 2 * out_dim)
                ops = nn.Sequential(
                    nn.ConvTranspose2d(in_dim, dim, 2, stride=2),
                    nn.GroupNorm(1, dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
                    nn.GroupNorm(1, dim // 2),
                    nn.Conv2d(dim // 2, out_dim, 1),
                    nn.GroupNorm(1, out_dim),
                    nn.GELU())
            elif k == 2:
                dim = max(in_dim // 2, out_dim)
                ops = nn.Sequential(
                    nn.ConvTranspose2d(in_dim, dim, 2, stride=2),
                    nn.GroupNorm(1, dim),
                    nn.Conv2d(dim, out_dim, 1),
                    nn.GroupNorm(1, out_dim),
                    nn.GELU())
            elif k == 1:
                ops = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1),
                    nn.GroupNorm(1, out_dim),
                    nn.GELU())
            elif k == 0.5:
                dim = max(2 * in_dim, out_dim)
                ops = nn.Sequential(
                    nn.Conv2d(in_dim, dim, 2, stride=2),
                    nn.GroupNorm(1, dim),
                    nn.Conv2d(dim, out_dim, 1),
                    nn.GroupNorm(1, out_dim),
                    nn.GELU())
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')
            self.rescale_ops.append(ops)

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales) or len(inputs) == 1
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in self.rescales]
        outputs = list()
        for i in range(len(inputs)):
            outputs.append(self.rescale_ops[i](inputs[i]))
        return tuple(outputs)


@BACKBONES.register_module()
class MAEWithSimpleFPN(MAE):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 num_fcs=2,
                 norm_eval=False,
                 pretrained=None,
                 init_values=0.1,
                 init_cfg=None,
                 fpn_cfg=dict(rescales=(4, 2, 1, 0.5),
                              out_dims=(64, 128, 256, 512)),
                 pos_embed_resize_mode='nearest',
                 rel_pos_embed_resize_mode='nearest',
                 use_rel_pos_embed=True):
        self.pos_embed_resize_mode = pos_embed_resize_mode
        self.rel_pos_embed_resize_mode = rel_pos_embed_resize_mode
        super(MAEWithSimpleFPN, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_indices=out_indices,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            patch_norm=patch_norm,
            final_norm=final_norm,
            num_fcs=num_fcs,
            norm_eval=norm_eval,
            pretrained=pretrained,
            init_values=init_values,
            init_cfg=init_cfg)
        fpn_cfg['in_dim'] = embed_dims
        fpn_cfg.setdefault('type', 'SimpleFPN')
        if isinstance(fpn_cfg, dict):
            self.fpn = build_neck(fpn_cfg)
            self.with_fpn = True
        else:
            self.with_fpn = False
        self.use_rel_pos_embed = use_rel_pos_embed
        if not use_rel_pos_embed:
            def delete_rel_pos_embed(module):
                if hasattr(module, 'relative_position_bias_table'):
                    del module.relative_position_bias_table
                    module.relative_position_bias_table = None
            self.apply(delete_rel_pos_embed)

    def _build_layers(self):
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, self.num_layers)
        ]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            attn_cfg = dict(
                rel_pos_embed_resize_mode=self.rel_pos_embed_resize_mode)
            self.layers.append(
                FlexSizeMAETransformerEncoderLayer(
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    feedforward_channels=self.mlp_ratio * self.embed_dims,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=self.num_fcs,
                    bias=True,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg,
                    window_size=self.patch_shape,
                    init_values=self.init_values,
                    attn_cfg=attn_cfg))

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)
        if self.pos_embed.size(1) == hw_shape[0] * hw_shape[1] + 1:
            pos_embed = self.pos_embed
        else:
            pos_embed = rearrange(
                self.pos_embed[:, 1:], 'b (h w) c -> b c h w',
                h=self.patch_shape[0], w=self.patch_shape[1])
            pos_embed = F.interpolate(
                pos_embed, hw_shape, mode=self.pos_embed_resize_mode)
            pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')
            pos_embed = torch.cat([self.pos_embed[:, :1], pos_embed], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        outs = tuple(outs)
        if self.with_fpn:
            return self.fpn(outs)
        return outs
