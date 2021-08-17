from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling.ops import batch_norm

__all__ = ['YOLOv2']

@register
class YOLOv2(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet19',
                 neck='YOLOv2Neck',
                 yolo_head='YOLOv3Head',
                 post_process='BBoxPostProcess',
                 num_classes = 20,
                 data_format='NCHW'):
        """
        YOLOv2 network
        """
        super(YOLOv2, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.num_classes = num_classes
 
    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        # kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'])

        # # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }
    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats)

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)

            return yolo_losses

        else:
            yolo_head_outs = self.yolo_head(neck_feats)

            bbox, bbox_num = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors,
                    self.inputs['im_shape'], self.inputs['scale_factor'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
