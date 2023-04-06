import torch
from torch.nn.functional import pad

from mmseg.apis.inference import LoadImage
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
from interformer import utils
from mmcv.utils.misc import to_2tuple

CLK_POSITIVE = 'positive'
CLK_NEGATIVE = 'negative'


class Predictor(object):
    def __init__(self, model, device, predictor_params):
        self.model = model
        self.device = device
        # params
        if predictor_params is not None:
            self.inner = predictor_params['inner_radius']
            self.outer = predictor_params['outer_radius']
            self.zoom_in_params = predictor_params['zoom_in_params']
        # data
        self.data = None
        self.image_feature = None
        self.img_ori_shape = None
        self.img_pad_shape = None

        # result
        self.prev_prediction = None
        self.ref_label = None

    def partially_reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.inner = predictor_params['inner_radius']
            self.outer = predictor_params['outer_radius']
            self.zoom_in_params = predictor_params['zoom_in_params']
        if self.data is not None:
            self.ref_label = utils.REF_UNKNOWN * torch.ones(size=(1, 1) + to_2tuple(self.img_pad_shape)).to(self.device)
            self.prev_prediction = torch.zeros((1, 1) + self.img_pad_shape).to(self.device)

    def set_input_image(self, img):
        cfg = self.model.cfg
        device = next(self.model.parameters()).device

        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)

        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = data['img_metas'].data[0]

        self.data = data
        self.image_feature = self.model.backbone(data['img'])
        self.img_ori_shape = data['img_metas'][0]["ori_shape"][:2]
        self.img_pad_shape = data['img_metas'][0]["pad_shape"][:2]

        self.ref_label = utils.REF_UNKNOWN * torch.ones(size=(1, 1) + to_2tuple(self.img_pad_shape)).to(self.device)
        self.prev_prediction = torch.zeros((1, 1) + self.img_pad_shape).to(self.device)

    def update_ref_label_by_new_click(self, click):
        y, x = click.coords
        mode = CLK_POSITIVE if click.is_positive else CLK_NEGATIVE
        self.ref_label = self.model.update_ref_label_by_point_lists(  # 根据更新后的point_lists更新ref_label
            ref_label=self.ref_label,
            point_lists=[[(y, x, mode)]],

            inner_radius=self.inner,
            outer_radius=self.outer)

    def get_prediction(self, num_click, prev_mask):
        if prev_mask is not None:
            prev_mask = pad(
                prev_mask,
                (0, self.img_pad_shape[1]-self.img_ori_shape[1], 0, self.img_pad_shape[0]-self.img_ori_shape[0]),
                "constant", 0
            )
            self.ref_label = self.model.update_ref_label_by_prediction(self.ref_label, prev_mask)
        roi =None
        if self.zoom_in_params is not None:
            if num_click > self.zoom_in_params["skip_clicks"]:
                roi = self.model.get_roi(
                    pre_label=self.prev_prediction,
                    ref_label=self.ref_label,
                    expand_ratio=(self.zoom_in_params["expansion_ratio"],)*2,
                    max_stride=max(self.img_pad_shape[-1] // i.size(-1) for i in self.image_feature),
                    min_size=self.zoom_in_params["min_size"])
        out = self.model.encode_decode(self.data['img'], self.data['img_metas'], x=self.image_feature, ref_label=self.ref_label, roi=roi)
        out = out.argmax(dim=1, keepdim=True)

        self.prev_prediction = out
        self.ref_label = self.model.update_ref_label_by_prediction(self.ref_label, out)
        return out.cpu().numpy()[0, 0, :self.img_ori_shape[0], :self.img_ori_shape[1]]

    def get_states(self):
        return {
            'prev_prediction': self.prev_prediction.clone(),
            'ref_label': self.ref_label.clone(),
        }

    def set_states(self, states):
        self.prev_prediction = states['prev_prediction']
        self.ref_label = states['ref_label']
