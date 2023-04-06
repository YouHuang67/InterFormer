import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.segmentors import EncoderDecoder


class BaseInterSegmentor(EncoderDecoder):

    def __init__(self,
                 *args,
                 num_stages=1,
                 **kwargs):
        """
        Args:
            *args: args to be passed to the parent class.
            num_stages (int): the number of stages.
            **kwargs: keyword args to be passed to the parent class.
        """
        self.num_stages = num_stages
        super(BaseInterSegmentor, self).__init__(*args, **kwargs)
        assert self.with_neck

    def _init_decode_head(self, decode_head):
        """
        Initialize the decode_head module.

        Args:
            decode_head (dict or list): the decode_head module.
        """
        if self.num_stages == 1:
            assert isinstance(decode_head, dict), \
                'decode_head must be a dict for num_stages == 1'
            super(BaseInterSegmentor, self)._init_decode_head(decode_head)
            return
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                img_metas,
                return_loss=True,
                interact=False,
                **kwargs):
        """
        Perform forward pass during training or inference.

        Args:
            img (Tensor): Input image tensor.
            img_metas (list[dict]): List of image information.
            return_loss (bool, optional): Whether to return losses. Default to True.
            interact (bool, optional): Whether to perform interactive inference. Default to False.

        Returns:
            dict: Result dict containing losses or output results.
        """
        if interact:
            return self.interact_test(img, img_metas, **kwargs)

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            # deploy forward_test in validation during training
            return self.forward_test(img, img_metas, **kwargs)

    @torch.no_grad()
    def interact_train(self, img, img_metas, gt_semantic_seg, **inter_info):
        """
        Perform interactive training.
        """
        raise NotImplementedError

    @torch.no_grad()
    def interact_test(self, img, img_metas, gt_semantic_seg, **inter_info):
        """
        Perform interactive testing.
        """
        raise NotImplementedError

    def forward_train(self, img, img_metas, gt_semantic_seg, **inter_info):
        """
        Forward function for training.

        Args:
            img (Tensor): Input image tensor.
            img_metas (list[dict]): List of image info dictionaries.
            gt_semantic_seg (Tensor): Ground truth semantic segmentation map.
            **inter_info: Other info for forward.

        Returns:
            dict: A dictionary of loss items.
        """
        self.eval()
        gt_semantic_seg, inter_info = self.interact_train(
            img, img_metas, gt_semantic_seg, **inter_info)
        self.train()
        x = self.backbone(img)
        x = self.neck(x, **inter_info)
        losses = dict()
        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def encode_decode(self, img, img_metas, x=None, size=None, **inter_info):
        """
        Encode the input image and decode it back to the output.

        Args:
            img (Tensor): Input image tensor.
            img_metas (list[dict]): List of image info dictionaries.
            x (Tensor): Input tensor, usually backbone feature map.
            size (tuple[int], optional): Desired size of output tensor.
            **inter_info: Other info for forward.

        Returns:
            Tensor: The output tensor.
        """
        if x is None:
            x = self.backbone(img)
        x = self.neck(x, **inter_info)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=(img.shape[2:] if size is None else size),
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Forward function for training with the decode head."""
        if self.num_stages == 1:
            return super(BaseInterSegmentor, self)._decode_head_forward_train(
                x, img_metas, gt_semantic_seg)

        losses = dict()
        loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            # Forward test again, maybe unnecessary for most methods.
            if i == 1:
                prev_outputs = self.decode_head[0].forward_test(
                    x, img_metas, self.test_cfg)
            else:
                prev_outputs = self.decode_head[i - 1].forward_test(
                    x, prev_outputs, img_metas, self.test_cfg)

            loss_decode = self.decode_head[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Forward function for testing with the decode head."""
        if self.num_stages == 1:
            return super(BaseInterSegmentor, self)._decode_head_forward_test(
                x, img_metas)

        seg_logits = self.decode_head[0].forward_test(
            x, img_metas, self.test_cfg)

        for i in range(1, self.num_stages):
            seg_logits = self.decode_head[i].forward_test(
                x, seg_logits, img_metas, self.test_cfg)

        return seg_logits

    def forward_test(self, img, img_meta, **inter_info):
        """
        Args:
            img (torch.Tensor): Input image tensor.
            img_meta (list[dict]): List of image information.
            **inter_info: Optional intermediate information.

        Returns:
            list: List of predicted segmentation maps.
        """
        # Ensure all images have the same original shape.
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        # Encode and decode the image to obtain the segmentation logits.
        seg_logit = self.encode_decode(img, img_meta, **inter_info)

        # Resize the segmentation logits to the original image size.
        resize_shape = img_meta[0]['img_shape'][:2]
        seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
        seg_logit = resize(seg_logit, size=img_meta[0]['ori_shape'][:2],
                           mode='bilinear', align_corners=self.align_corners,
                           warning=False)

        # Obtain the predicted segmentation map.
        output = F.softmax(seg_logit, dim=1)
        if img_meta[0]['flip']:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))
        seg_pred = output.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)

        return seg_pred
