import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, ReLU
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ipdb import set_trace
from tools.mambaout_en import GatedCNNBlock, DownsampleLayer, UpsampleLayer, StemLayer
from tools.fusion_transformer import Fusion_network_Transformer

from configs.TFB_1 import args
from functools import partial


class SplAtConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0):
        super(SplAtConv2d, self).__init__()
        self.rectify = rectify
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation, groups=groups*radix, bias=bias)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.radix > 1:
            splited = torch.split(x, self.channels//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(x.shape[0], -1, 1, 1)
        if self.radix > 1:
            attens = torch.split(atten, self.channels//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class ResNeStBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False, rectified_conv=False, rectify_avg=False, norm_layer=BatchNorm2d, dropblock_prob=0.0, last_gamma=False):
        super(ResNeStBlock, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        self.conv2 = SplAtConv2d(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=dilation, groups=cardinality, bias=False,
            radix=radix, rectify=rectified_conv,
            rectify_avg=rectify_avg,
            norm_layer=norm_layer,
            dropblock_prob=dropblock_prob)
        self.conv3 = Conv2d(
            group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*self.expansion)
        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DownsampleLayer, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class ResNeStLayer(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, stride=1):
        super(ResNeStLayer, self).__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample=downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

# Dense Block unit
# light version
class GAP2D(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)


# NestFuse network - light, no desnse
class NestFuse_light2_nodense(nn.Module):
    
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(NestFuse_light2_nodense, self).__init__()
        print(f"NEST: init2")
        self.deepsupervision = deepsupervision
        self.output_filter = output_filter = 16
        kernel_size = 3
        stride = 1
        self.pool = nn.MaxPool2d(2, 2)
        # dims=[128, 256, 512, 768]
        dims = nb_filter
        # downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        # encoder
        # self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        # self.conv_dem = ConvLayer(1, output_filter, 1, stride)
        # self.conv_rgb = StemLayer(in_channels=input_nc, out_channels=dims[0])
        # self.conv_dem = StemLayer(in_channels=output_nc, out_channels=dims[0]) ################################
        
        num_layers = [3, 4, 6, 3]  # Example number of layers

        # encoder
        self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        self.conv_dem = ConvLayer(output_nc, output_filter, 1, stride)
        
        # self.EB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        # self.EB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        # # self.EB3_0 = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        # self.EB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        # self.EB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        self.RB1_0 = ResNeStLayer(ResNeStBlock, output_filter, nb_filter[0], num_layers[0])

        self.RB2_0 = nn.Sequential(
            # DownsampleLayer(nb_filter[0], nb_filter[1]),
            ResNeStLayer(ResNeStBlock, nb_filter[0],nb_filter[1], num_layers[1])
        )
        self.RB3_0 = nn.Sequential(
            # DownsampleLayer(nb_filter[1], nb_filter[2]),
            ResNeStLayer(ResNeStBlock, nb_filter[1], nb_filter[2], num_layers[2])
        )
        self.RB4_0 = nn.Sequential(
            # DownsampleLayer(nb_filter[2], nb_filter[3]),
            ResNeStLayer(ResNeStBlock, nb_filter[2], nb_filter[3], num_layers[3])
        )

        self.RB1_dem = ResNeStLayer(ResNeStBlock, output_filter, nb_filter[0], num_layers[0])

        self.RB2_dem = nn.Sequential(
            # DownsampleLayer(nb_filter[0], nb_filter[1]),
            ResNeStLayer(ResNeStBlock, nb_filter[0], nb_filter[1], num_layers[1])
        )
        self.RB3_dem = nn.Sequential(
            # DownsampleLayer(nb_filter[1], nb_filter[2]),
            ResNeStLayer(ResNeStBlock, nb_filter[1], nb_filter[2], num_layers[2])
        )
        self.RB4_dem = nn.Sequential(
            # DownsampleLayer(nb_filter[2], nb_filter[3]),
            ResNeStLayer(ResNeStBlock, nb_filter[2], nb_filter[3], num_layers[3])
        )

        # decoder
        # self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        '''
        self.DB1 = nn.Sequential(self._make_layer(dims[0], num_layers[0]),DownsampleLayer(dims[0], dims[1]))
        self.down1 = DownsampleLayer(dims[0], dims[1])
        self.DB2 = nn.Sequential(self._make_layer(dims[1], num_layers[1]))
        self.down2 = DownsampleLayer(dims[1], dims[2])
        self.DB3 = nn.Sequential(self._make_layer(dims[2], num_layers[2]))
        self.down3 = DownsampleLayer(dims[2], dims[3])
        self.DB4 = nn.Sequential(self._make_layer(dims[3], num_layers[3]))
        
        self.DB1_dem = nn.Sequential(self._make_layer(dims[0], num_layers[0]),DownsampleLayer(dims[0], dims[1]))
        self.DB2_dem = nn.Sequential(self._make_layer(dims[1], num_layers[1]))
        self.DB3_dem = nn.Sequential(self._make_layer(dims[2], num_layers[2]))
        self.DB4_dem = nn.Sequential(self._make_layer(dims[3], num_layers[3]))
        '''
        self.down1 = DownsampleLayer(nb_filter[0], nb_filter[1])
        self.down2 = DownsampleLayer(nb_filter[1], nb_filter[2])
        self.down3 = DownsampleLayer(nb_filter[2], nb_filter[3])
        self.funsion = Fusion_network_Transformer(nb_filter)
        self.gap = GAP2D()
        
        self.fc = nn.Linear(dims[3], 1)
        
    def forward(self, input):
        # print(f"NEST:{input.shape}")
        # print(f"NEST:{input[:,3:,:,:].shape}")

        x = self.conv_rgb(input[:,:3,:,:])
        x_dem = self.conv_dem(input[:,3:,:,:])
        # x = x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        # set_trace()
        # torch.Size([2, 16, 128, 128])
        x1_0 = self.RB1_0(x)
        # ([2, 64, 128, 128])

        x2_0 = self.RB2_0(x1_0)
        # ([2, 112, 64, 64])

        x3_0 = self.RB3_0(x2_0)
        # ([2, 160, 32, 32])

        x4_0 = self.RB4_0(x3_0)
        # ([2, 208, 16, 16])
        # x1_0 = x1_0.permute(0, 3, 1, 2)
        # x2_0 = x2_0.permute(0, 3, 1, 2)
        # x3_0 = x3_0.permute(0, 3, 1, 2)
        # x4_0 = x4_0.permute(0, 3, 1, 2)
        
        x1_dem = self.RB1_dem(x_dem)
        # ([2, 64, 128, 128])

        x2_dem = self.RB2_dem(x1_dem)
        # ([2, 112, 64, 64])

        x3_dem = self.RB3_dem(x2_dem)
        # ([2, 160, 32, 32])

        x4_dem = self.RB4_dem(x3_dem)
        # x1_dem = x1_dem.permute(0, 3, 1, 2)
        # x2_dem = x2_dem.permute(0, 3, 1, 2)
        # x3_dem = x3_dem.permute(0, 3, 1, 2)
        # x4_dem = x4_dem.permute(0, 3, 1, 2)
        f_en = self.funsion([x1_0, x2_0, x3_0, x4_0],[x1_dem, x2_dem, x3_dem, x4_dem])
        f_en[0] = f_en[0] + x1_0 + x1_dem
        f_en[1] = f_en[1] + x2_0 + x2_dem
        f_en[2] = f_en[2] + x3_0 + x3_dem
        f_en[3] = f_en[3] + x4_0 + x4_dem
        # torch.Size([4, 64, 256, 256])
        # torch.Size([4, 112, 128, 128])
        # torch.Size([4, 160, 64, 64])
        # torch.Size([4, 208, 32, 32])
        x1 = self.down1(f_en[0].permute(0, 2, 3, 1))

        x2 = f_en[1].permute(0, 2, 3, 1)
        x2 = x1 + x2
        # x2_output = x2.permute(0, 3, 1, 2)
        x2 = self.down2(x2)

        x3 = f_en[2].permute(0, 2, 3, 1)
        # set_trace()
        x3 = x2 + x3
        # x3_output = x3.permute(0, 3, 1, 2)
        x3 = self.down3(x3)

        x4 = f_en[3].permute(0, 2, 3, 1)
        x4 = (x3 + x4).permute(0, 3, 1, 2)
        # set_trace()
        # output1 = self.conv_out(x1_3)
        # 这里的维度应该会有问题
        output1 = self.gap(x4)
        # out = self.fc_l(x1_3.view(-1, self.output_filter * 128 * 128))
        out = self.fc(output1)
        # out = self.fc(output1.view(-1, self.output_filter * 128 * 128))
        return out