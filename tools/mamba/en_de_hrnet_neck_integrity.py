import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ipdb import set_trace
from tools.mambaout_en import GatedCNNBlock, DownsampleLayer, UpsampleLayer, StemLayer
from tools.fusion_transformer import Fusion_network_Transformer
from CrossFuse.network.transformer_cam import cross_encoder
from configs.TFB_1 import args
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAttention(nn.Module):
    def __init__(self, in_channels, channels, cardinality, reduction_ratio=16):
        super(SplitAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.cardinality = cardinality
        self.reduction_ratio = reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y).view(b, c, -1)
        y = self.sigmoid(y).view(b, c, self.cardinality, -1)
        return x * y.expand_as(x)

class ResNeStBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, reduction_ratio=16):
        super(ResNeStBlock, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=kernel_size, stride=stride, groups=out_channels//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//2)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.split_attention = SplitAttention(out_channels, out_channels//4, groups, reduction_ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.split_attention(out)

        out += residual
        out = F.relu(out)

        return out

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
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

# NestFuse network - light, no desnse
class NestFuse_light2_nodense(nn.Module):
    
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(NestFuse_light2_nodense, self).__init__()
        print(f"NEST: init2")
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
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
        
        num_layers = args.mamba_num_layers

        # encoder
        self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        self.conv_dem = ConvLayer(output_nc, output_filter, 1, stride)
        
        self.EB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        # self.EB3_0 = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        self.EB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)


        self.EB1_dem = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_dem = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_dem = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        # self.EB3_dem = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        self.EB4_dem = block(nb_filter[2], nb_filter[3], kernel_size, 1)
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
        x1_0 = self.EB1_0(x)
        # ([2, 64, 128, 128])

        x2_0 = self.EB2_0(self.pool(x1_0))
        # ([2, 112, 64, 64])

        x3_0 = self.EB3_0(self.pool(x2_0))
        # ([2, 160, 32, 32])

        x4_0 = self.EB4_0(self.pool(x3_0))
        # ([2, 208, 16, 16])
        # x1_0 = x1_0.permute(0, 3, 1, 2)
        # x2_0 = x2_0.permute(0, 3, 1, 2)
        # x3_0 = x3_0.permute(0, 3, 1, 2)
        # x4_0 = x4_0.permute(0, 3, 1, 2)
        
        x1_dem = self.EB1_dem(x_dem)
        # ([2, 64, 128, 128])

        x2_dem = self.EB2_dem(self.pool(x1_dem))
        # ([2, 112, 64, 64])

        x3_dem = self.EB3_dem(self.pool(x2_dem))
        # ([2, 160, 32, 32])

        x4_dem = self.EB4_dem(self.pool(x3_dem))
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
    

class Conv_crossfuse(nn.Module):
    
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(Conv_crossfuse, self).__init__()
        print(f"NEST: init2")
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
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
        

        # encoder
        self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        self.conv_dem = ConvLayer(output_nc, output_filter, 1, stride)
        
        self.EB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        # self.EB3_0 = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        self.EB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)


        self.EB1_dem = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_dem = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_dem = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        # self.EB3_dem = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        self.EB4_dem = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        self.down1 = DownsampleLayer(nb_filter[0], nb_filter[1])
        self.down2 = DownsampleLayer(nb_filter[1], nb_filter[2])
        self.down3 = DownsampleLayer(nb_filter[2], nb_filter[3])
        self.img_size = 32  # 32*32
        self.patch_size = 2  # 2*2
        self.part_out = 160  # 128
        self.embed_dim = self.part_out * self.patch_size * self.patch_size  # 512
        self.num_patches = int(self.img_size / self.patch_size) * int(self.img_size / self.patch_size)  # 16*16

        self.cross_atten_block = cross_encoder(self.img_size, self.patch_size, self.embed_dim, self.num_patches, depth_self=1,
                                               depth_cross=1)
        
        self.gap = GAP2D()
        
        self.fc = nn.Linear(dims[2]*3, 1)
        
    def forward(self, input):
        # print(f"NEST:{input.shape}")
        # print(f"NEST:{input[:,3:,:,:].shape}")

        x = self.conv_rgb(input[:,:3,:,:])
        x_dem = self.conv_dem(input[:,3:,:,:])
        # x = x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        # set_trace()
        # torch.Size([2, 16, 128, 128])
        x1_0 = self.EB1_0(x)
        # ([2, 64, 128, 128])

        x2_0 = self.EB2_0(self.pool(x1_0))
        # ([2, 112, 64, 64])

        x3_0 = self.EB3_0(self.pool(x2_0))
        # ([2, 160, 32, 32])

        x4_0 = self.EB4_0(self.pool(x3_0))
        # ([2, 208, 16, 16])
        # x1_0 = x1_0.permute(0, 3, 1, 2)
        # x2_0 = x2_0.permute(0, 3, 1, 2)
        # x3_0 = x3_0.permute(0, 3, 1, 2)
        # x4_0 = x4_0.permute(0, 3, 1, 2)
        
        x1_dem = self.EB1_dem(x_dem)
        # ([2, 64, 128, 128])

        x2_dem = self.EB2_dem(self.pool(x1_dem))
        # ([2, 112, 64, 64])

        x3_dem = self.EB3_dem(self.pool(x2_dem))
        # ([2, 160, 32, 32])

        x4_dem = self.EB4_dem(self.pool(x3_dem))
        # x1_dem = x1_dem.permute(0, 3, 1, 2)
        # x2_dem = x2_dem.permute(0, 3, 1, 2)
        # x3_dem = x3_dem.permute(0, 3, 1, 2)
        # x4_dem = x4_dem.permute(0, 3, 1, 2)
        
        fusion = self.cross_atten_block(x3_0, x3_dem)
        stack = torch.cat((fusion, x3_0, x3_dem), dim=1)
        
        # 这里的维度应该会有问题
        output1 = self.gap(stack)
        # out = self.fc_l(x1_3.view(-1, self.output_filter * 128 * 128))
        out = self.fc(output1)
        # out = self.fc(output1.view(-1, self.output_filter * 128 * 128))
        return out
    
