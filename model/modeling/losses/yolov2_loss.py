# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from model.core.workspace import register

from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity

__all__ = ['YOLOv2Loss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv2Loss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        
        super(YOLOv2Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        # ?????????[N, 3, H, W, 4]
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        # ?????????[N, M1, 4]???M1?????????????????????
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        # ?????????[n, M2, 4]???M2?????????????????????
        gbox = paddle.concat([gxy, gwh], axis=-1)

        # iou?????????[N, M1, M2]
        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]     # ??????????????????iou????????????????????????
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype) # iou??????????????????????????????????????????????????????
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        # loss_obj = F.binary_cross_entropy_with_logits(
        #     pobj, obj_mask, reduction='none')
        loss_obj = paddle.pow(F.sigmoid(pobj)-obj_mask, 2)
        loss_obj_pos = (loss_obj * tobj) * 5   # ??????????????????objectness_loss???
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        
        # loss_cls = F.binary_cross_entropy_with_logits(
        #     pcls, tcls, reduction='none')
        loss_cls = paddle.pow(F.sigmoid(pcls)-tcls, 2)
        return loss_cls

    # ???????????????????????????loss
    def yolov2_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        # ?????????NCHW?????????????????????????????????[N, 3, 85, H, W]??????????????????[N, 3, H, W, 85]   ?????????COCO??????????????????iou_aware??????
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        # ????????????????????????????????????????????????objectness???????????????
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        # ?????????targets??????????????????
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        #tscale_obj = tscale * tobj 
        tscale_obj = tscale * tobj - tobj * 0.01 + 0.01
        loss = dict()

        # ???????????????????????????????????????????????????????????????????????????offset)
             # ???scale=2??????yolov5????????????????????????????????????????????????????????????????????????????????????GT2Targets??????????????????
                # ppyolo(v2)??????yolov4???scale???1.05????????????grid sensitive
        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        loss_x = paddle.pow(x - tx, 2)               
        loss_y = paddle.pow(y - ty, 2)
        loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()    
        loss_w = paddle.pow(w - tw, 2)
        loss_h = paddle.pow(h - th, 2)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        box = [x, y, w, h]    
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)  
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()    
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
                                            self.downsample):
            yolo_loss = self.yolov2_loss(x, t, gt_box, anchor, downsample,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses
