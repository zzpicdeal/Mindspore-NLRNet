
import mindspore 
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import numpy as np 


class ContextBlock2d(nn.Cell):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=8):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)  # context Modeling
            self.softmax = nn.Softmax(axis=2)
            self.avg_pool = ops.AdaptiveAvgPool2D(1)
        else:
            self.avg_pool = ops.AdaptiveAvgPool2D(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.SequentialCell(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1,weight_init = 'Zero')
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1,weight_init = 'Zero')
            )
        else:
            self.channel_mul_conv = None

        self.conv = nn.Conv1d(1, 1, kernel_size=9, pad_mode='pad',padding=(9 - 1) // 2, has_bias=False)
        # self.conv2 = nn.Conv1d(1, 1, kernel_size=9, padding=(9 - 1) // 2, bias=False)
        #         self.relu = nn.LeakyReLU(0.2, inplace=True)



        self.gamma = Parameter(Tensor(np.zeros(1),mindspore.float32),name='gamma')
        self.sig = ops.Sigmoid()
        self.unsq = ops.ExpandDims()

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.reshape(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = self.unsq(input_x,1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.reshape(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)  # softmax操作
            # [N, 1, H * W, 1]
            context_mask = self.unsq(context_mask,3)#.unsqueeze(3)
            # [N, 1, C, 1]
            context = ops.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.reshape(batch, channel, 1, 1)

            # novel:
            # context2 = self.avg_pool(x)
            # context = torch.cat((context, context2), dim=1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def construct(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = self.sig(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            # channel_add_term = self.channel_add_conv(context)
            channel_add_term = self.unsq(self.conv(context.squeeze(-1).transpose(0,2,1)).transpose(0,2,1),2)#.unsqueeze(-1)
            # channel_add_term = self.conv2(self.relu(channel_add_term)).transpose(-1, -2).unsqueeze(-1)
            out = out + self.gamma * channel_add_term
            # out = out + channel_add_term

        return out


class _Residual_Block(nn.Cell):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='pad',padding=1, has_bias=False)
        self.in1 = nn.GroupNorm(64,64)#nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='pad',padding=1, has_bias=False)
        self.in2 = nn.GroupNorm(64,64)#nn.InstanceNorm2d(64, affine=True)

        self.gc = ContextBlock2d(64, 64)
        self.gamma = Parameter(Tensor(np.zeros(1),mindspore.float32),name='gamma')

    def construct(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = self.gamma * output + identity_data
        output = self.gc(output)
        return output

class NLRNet(nn.Cell):
    def __init__(self, dataset='gf2'):
        super(NLRNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv_input = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=1, pad_mode='pad',padding=3, has_bias=False,weight_init = mindspore.common.initializer.Normal(sigma=0.025, mean=0.0))
        #self.residual = self.make_layer(_Residual_Block, 16)
        self.residual = self.make_layer(_Residual_Block, 8)
        #self.residual = self.make_layer(_Residual_Block, 24)
        #self.residual = self.make_layer(_Residual_Block, 12)
        #self.residual = self.make_layer(_Residual_Block, 20)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='pad',padding=1, has_bias=False,weight_init = mindspore.common.initializer.Normal(sigma=0.058, mean=0.0))
        self.bn_mid = nn.GroupNorm(64,64)#nn.InstanceNorm2d(64, affine=True)
        self.conv_output = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, pad_mode='same',padding=0, has_bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, pad_mode='same',padding=0, has_bias=False),
        )

        # 无用模块
        self.outconv = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, pad_mode='pad',padding=1, has_bias=False)

        self.cat = ops.Concat(1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.SequentialCell(*layers)

    def construct(self, *input):
        ms, pan = input[0], input[3]
        
        detail = self.cat((ms, pan))
        out = self.lrelu(self.conv_input(detail))

        residual = out
        out = self.residual(out)

        out = self.bn_mid(self.conv_mid(out))
        out = ops.mul(out,residual)
        out = self.conv_output(out)
        return out

