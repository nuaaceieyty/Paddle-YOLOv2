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
from ppdet.core.workspace import register

from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity

__all__ = ['YOLOv3Loss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv3Loss(nn.Layer):

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
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOv3Loss, self).__init__()
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
        # 形状为[N, 3, H, W, 4]
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        # 形状为[N, M1, 4]；M1为预测框的个数
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        # 形状为[n, M2, 4]；M2为真实框的个数
        gbox = paddle.concat([gxy, gwh], axis=-1)

        # iou形状为[N, M1, M2]
        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]     # 找到与真实框iou小于阈值的预测框
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype) # iou大于阈值的非匹配检测框为非正非负样本
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)    # 对正样本计算objectness_loss，
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    # 用于计算某一分支的loss
    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        # 将原本NCHW形式的特征图转换为形状[N, 3, 85, H, W]，再转为形状[N, 3, H, W, 85]   （对于COCO数据集且没用iou_aware时）
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        # 从特征图中解析出中心坐标，宽高，objectness，类别分数
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        # 解析出targets中的相应信息
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()

        # 这里对特征图中的相关值进行计算，得到中心位置坐标（offset)
             # 当scale=2时与yolov5一致，相当于一个格点可以预测中心位置在临近格点的目标（但GT2Targets要对应写好）
                # ppyolo(v2)中和yolov4中scale取1.05，作用为grid sensitive
        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            # 因为x,y之前已经做过转换计算，所以这里不需要with_logits
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')   # 交叉熵损失
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)               # L1 loss
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()    # 在其他维度求和后，在batch维度取平均
                                                        # 即输入尺寸越大则此loss也会相应变大，batch_size对此loss影响不大
        loss_w = paddle.abs(w - tw)
        loss_h = paddle.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            # 分别求出预测框与真实框的中心位置、宽高，这里是归一化后的值
               # 对于没有目标对应的格点（锚框），相应求出的“真实框”即为中心位于格点中心的锚框本身
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj          # 此步是为了使得只计算有目标对应的部分的loss
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()  # 在其他维度求和后，在batch维度取平均
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, w, h]    # x,y为已经计算完的offset，w,h还未进行与锚框的计算
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)  # 计算objectness部分的loss
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj     # 只在有目标的位置计算objectness的损失函数
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()    # 在其他维度求和后，在batch维度取平均
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
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample,
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
