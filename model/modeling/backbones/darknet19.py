
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from model.core.workspace import register, serializable
from model.modeling.ops import batch_norm, mish
from ..shape_spec import ShapeSpec

__all__ = ['DarkNet19', 'ConvBNLayer']

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 norm_type='bn',
                 norm_decay=0.,
                 act="leaky",
                 freeze_norm=False,
                 data_format='NCHW',
                 name=''):
        """
        conv + bn + activation layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 1
            groups (int): number of groups of conv layer, default 1
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            act (str): activation function type, default 'leaky', which means leaky_relu
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            data_format=data_format,
            bias_attr=False)
        self.batch_norm = batch_norm(
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        elif self.act == 'mish':
            out = mish(out)
        return out

@register
@serializable
class DarkNet19(nn.Layer):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 norm_type='bn',
                 norm_decay=0.,
                 data_format='NCHW'):
        """
        Darknet19

        """
        super(DarkNet19, self).__init__()
        self.conv0 = ConvBNLayer(ch_in=3, ch_out=32, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2) 
        self.conv2 = ConvBNLayer(ch_in=32, ch_out=64, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2) 
        self.conv4 = ConvBNLayer(ch_in=64, ch_out=128, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv5 = ConvBNLayer(ch_in=128, ch_out=64, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv6 = ConvBNLayer(ch_in=64, ch_out=128, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.pool7 = nn.MaxPool2D(kernel_size=2, stride=2) 
        self.conv8 = ConvBNLayer(ch_in=128, ch_out=256, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv9 = ConvBNLayer(ch_in=256, ch_out=128, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv10 = ConvBNLayer(ch_in=128, ch_out=256, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.pool11 = nn.MaxPool2D(kernel_size=2, stride=2) 
        self.conv12 = ConvBNLayer(ch_in=256, ch_out=512, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv13 = ConvBNLayer(ch_in=512, ch_out=256, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv14 = ConvBNLayer(ch_in=256, ch_out=512, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv15 = ConvBNLayer(ch_in=512, ch_out=256, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv16 = ConvBNLayer(ch_in=256, ch_out=512, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.pool17 = nn.MaxPool2D(kernel_size=2, stride=2) 
        self.conv18 = ConvBNLayer(ch_in=512, ch_out=1024, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv19 = ConvBNLayer(ch_in=1024, ch_out=512, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv20 = ConvBNLayer(ch_in=512, ch_out=1024, filter_size=3, stride=1,
                 padding=1, act='leaky')
        self.conv21 = ConvBNLayer(ch_in=1024, ch_out=512, filter_size=1, stride=1,
                 padding=0, act='leaky')
        self.conv22 = ConvBNLayer(ch_in=512, ch_out=1024, filter_size=3, stride=1,
                 padding=1, act='leaky')

    def forward(self, inputs):
        x = inputs['image']        
        x = self.conv0(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        shortcut = x
        x = self.pool17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        return [x, shortcut]

