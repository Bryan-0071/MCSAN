import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tools.fusion_transformer import Fusion_network_Transformer
from CrossFuse.network.transformer_cam import cross_encoder
from ipdb import set_trace
# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if not self.is_last:
            # out = F.normalize(out)
            out = self.bn(out)
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

class DenseBlock_strong(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_strong, self).__init__()
        mid_channels = out_channels
        self.conv1 = ConvLayer(in_channels, mid_channels, kernel_size, stride)
        self.conv2 = ConvLayer(mid_channels, mid_channels, kernel_size, stride)
        # Use a linear projection in the last layer without an internal ReLU.
        self.conv3 = ConvLayer(mid_channels, out_channels, 1, stride, is_last=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # return out
        return F.relu(out + identity)

class LambdaLayer1(nn.Module):
    def __init__(self, te):
        super(LambdaLayer1,self).__init__()
        self.te = te
    def forward(self, x):
        return x
# NestFuse network - light, no desnse
class NestFuse_light2_nodense2(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(NestFuse_light2_nodense2, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        # block = DenseBlock_strong
        self.output_filter = output_filter = 16
        kernel_size = 3
        stride = 1
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_four = nn.MaxPool2d(4, 4)
        self.up = nn.Upsample(scale_factor=2)
        # self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        self.conv_dem = ConvLayer(output_nc, output_filter, 1, stride)
        
        self.EB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_0 = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        # self.EB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        # self.EB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)


        self.EB1_dem = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_dem = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        # self.EB3_dem = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB3_dem = block(nb_filter[1]+nb_filter[0], nb_filter[2], kernel_size, 1)
        # self.EB4_dem = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        # self.funsion = Fusion_network_Transformer(nb_filter)
        self.img_size = 32  # 32*32
        self.patch_size = 2  # 2*2
        self.part_out = nb_filter[2]  # 128
        self.embed_dim = self.part_out * self.patch_size * self.patch_size  # 512
        self.num_patches = int(self.img_size / self.patch_size) * int(self.img_size / self.patch_size)  # 16*16

        self.cross_atten_block = cross_encoder(self.img_size, self.patch_size, self.embed_dim, self.num_patches, depth_self=1,
                                               depth_cross=1)
        self.lambdaLayer = LambdaLayer1(1)
        self.gap = GAP2D()
        self.fc = nn.Linear(nb_filter[2]*3, 2)

    def forward(self, images):
        # print(input.shape)
        C = images.shape[1]

        if C > 10:
            # Sentinel-2 full-band input in the common channel order.
            # print("Using Sentinel-2 full bands (12 channels)")
            rgb_idx = [3,2,1]   # B4, B3, B2
            dem_idx = [12,13]
        else:
            # bijie dataset
            rgb_idx = [0, 1, 2]
            dem_idx = [3, 4]
        input1 = images[:,rgb_idx,:,:]
        input2 = images[:,dem_idx,:,:]
        x = self.conv_rgb(input1)
        x_dem = self.conv_dem(input2)
        # set_trace()
        # torch.Size([2, 16, 128, 128])
        x1_0 = self.EB1_0(x)
        # ([2, 64, 128, 128])

        x2_0 = self.EB2_0(self.pool(x1_0))
        # ([2, 112, 64, 64])
        x3_0 = self.EB3_0(torch.cat((self.pool(x2_0), self.pool_four(x1_0)), dim=1))
        # x3_0 = self.EB3_0(self.pool(x2_0))
        # ([2, 160, 32, 32])

        # x4_0 = self.EB4_0(self.pool(x3_0))
        # ([2, 208, 16, 16])

        x1_dem = self.EB1_dem(x_dem)
        # ([2, 64, 128, 128])

        x2_dem = self.EB2_dem(self.pool(x1_dem))
        # ([2, 112, 64, 64])

        x3_dem = self.EB3_dem(torch.cat((self.pool(x2_dem), self.pool_four(x1_dem)), dim=1))
        # x3_dem = self.EB3_dem(self.pool(x2_dem))
        
        # ([2, 160, 32, 32])

        # x4_dem = self.EB4_dem(self.pool(x3_dem))
        # ([2, 208, 16, 16])
        # set_trace()
        # print("x3_0.shape:", x3_0.shape)
        # x3_0.shape: torch.Size([64, 256, 32, 32])
        fusion = self.cross_atten_block(x3_0, x3_dem,shift_flag=False)
        # fusion = self.cross_atten_block(x2_0, x2_dem)
        stack = torch.cat((fusion, x3_0, x3_dem), dim=1)
        # stack = torch.cat((x3_0, x3_dem), dim=1)
        stack = self.lambdaLayer(stack)
        output1 = self.gap(stack)
        # out = self.fc_l(x1_3.view(-1, self.output_filter * 128 * 128))
        out = self.fc(output1)
        
        # out = self.fc(output1.view(-1, self.output_filter * 128 * 128))
        return out
