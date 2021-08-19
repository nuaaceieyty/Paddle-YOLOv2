
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from model.core.workspace import register, serializable
from ..backbones.darknet import ConvBNLayer
from ..shape_spec import ShapeSpec

__all__ = ['YOLOv2Neck']

def reorg(x):
    x1 = x[:, :, 0::2, 0::2]
    x2 = x[:, :, 0::2, 1::2]
    x3 = x[:, :, 1::2, 0::2]
    x4 = x[:, :, 1::2, 1::2]
    return paddle.concat([x1, x2, x3, x4], axis=1)

@register
@serializable
class YOLOv2Neck(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 num_classes=20,
                 norm_type='bn',
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        YOLOv2Neck layer
        """
        super(YOLOv2Neck, self).__init__()
        self.num_classes = num_classes

        self.conv23 = ConvBNLayer(ch_in=1024, ch_out=1024, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv24 = ConvBNLayer(ch_in=1024, ch_out=1024, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv26 = ConvBNLayer(ch_in=512, ch_out=64, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv29 = ConvBNLayer(ch_in=1280, ch_out=1024, filter_size=1, stride=1,
                 padding=0, act='leaky')

    def forward(self, blocks):
        x, shortcut = blocks[0], blocks[1]

        x = self.conv23(x)
        x = self.conv24(x)
        shortcut = self.conv26(shortcut)
        reorg_ = reorg(shortcut)
        concat_ = paddle.concat([x, reorg_], axis=1)
        x = self.conv29(concat_)
        neck_feats = [x]

        return neck_feats

    @property
    def out_shape(self):
        return [ShapeSpec(channels=1024)]
