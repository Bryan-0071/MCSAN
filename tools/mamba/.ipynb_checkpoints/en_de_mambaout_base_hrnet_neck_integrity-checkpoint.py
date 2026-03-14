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
from configs.TFB_1 import args
from functools import partial
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
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()
        # dims=[128, 256, 512, 768]
        dims = nb_filter
        # downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        # encoder
        # self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        # self.conv_dem = ConvLayer(5, output_filter, 1, stride)
        self.conv_rgb = StemLayer(in_channels=3, out_channels=dims[0])
        self.conv_dem = StemLayer(in_channels=1, out_channels=dims[0]) ################################
        
        num_layers = args.mamba_num_layers

        #set_trace()
        self.EB1_0 = self._make_layer(dims[0], num_layers[0])
        
        self.EB2_0 = nn.Sequential(
            DownsampleLayer(dims[0], dims[1]),
            self._make_layer(dims[1], num_layers[1])
        )

        self.EB3_0 = nn.Sequential(
            DownsampleLayer(dims[1], dims[2]),
            self._make_layer(dims[2], num_layers[2])
        )
        
        self.EB4_0 = nn.Sequential(
            DownsampleLayer(dims[2], dims[3]),
            self._make_layer(dims[3], num_layers[3])
        )

        self.EB1_dem = self._make_layer(dims[0], num_layers[0])
        
        self.EB2_dem = nn.Sequential(
            DownsampleLayer(dims[0], dims[1]),
            self._make_layer(dims[1], num_layers[1])
        )

        self.EB3_dem = nn.Sequential(
            DownsampleLayer(dims[1], dims[2]),
            self._make_layer(dims[2], num_layers[2])
        )
        
        self.EB4_dem = nn.Sequential(
            DownsampleLayer(dims[2], dims[3]),
            self._make_layer(dims[3], num_layers[3])
        )
        # decoder
        # self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
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

        self.funsion = Fusion_network_Transformer(nb_filter)
        self.gap = GAP2D()
        
        self.fc = nn.Linear(dims[3], 2) 
        
        self.fc_l = nn.Linear(output_filter * 128 * 128, 2)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_filter, 1, stride)
    def _make_layer(self, channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(GatedCNNBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        print(f"NEST:{input.shape}")
        print(f"NEST:{input[:,3:,:,:].shape}")

        x = self.conv_rgb(input[:,:3,:,:])
        x_dem = self.conv_dem(input[:,3:,:,:])
        # x = x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        # set_trace()
        # torch.Size([2, 16, 128, 128])
        x1_0 = self.EB1_0(x)
        # ([2, 64, 128, 128])

        x2_0 = self.EB2_0(x1_0)
        # ([2, 112, 64, 64])

        x3_0 = self.EB3_0(x2_0)
        # ([2, 160, 32, 32])

        x4_0 = self.EB4_0(x3_0)
        # ([2, 208, 16, 16])
        x1_0 = x1_0.permute(0, 3, 1, 2)
        x2_0 = x2_0.permute(0, 3, 1, 2)
        x3_0 = x3_0.permute(0, 3, 1, 2)
        x4_0 = x4_0.permute(0, 3, 1, 2)
        
        x1_dem = self.EB1_dem(x_dem)
        # ([2, 64, 128, 128])

        x2_dem = self.EB2_dem(x1_dem)
        # ([2, 112, 64, 64])

        x3_dem = self.EB3_dem(x2_dem)
        # ([2, 160, 32, 32])

        x4_dem = self.EB4_dem(x3_dem)
        x1_dem = x1_dem.permute(0, 3, 1, 2)
        x2_dem = x2_dem.permute(0, 3, 1, 2)
        x3_dem = x3_dem.permute(0, 3, 1, 2)
        x4_dem = x4_dem.permute(0, 3, 1, 2)
        f_en = self.funsion([x1_0, x2_0, x3_0, x4_0],[x1_dem, x2_dem, x3_dem, x4_dem])

        x1 = self.down1(f_en[0].permute(0, 2, 3, 1))

        x2 = f_en[1].permute(0, 2, 3, 1)
        x2 = x1 + x2
        x2 = self.down2(x2)

        x3 = f_en[2].permute(0, 2, 3, 1)
        # set_trace()
        x3 = x2 + x3
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
    def decoder(self, f_en):
        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))
        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x1 = self.down1(f_en[0].permute(0, 2, 3, 1))

        x2 = f_en[1].permute(0, 2, 3, 1)
        x2 = x1 + x2
        x2 = self.down2(x2)

        x3 = f_en[2].permute(0, 2, 3, 1)
        # set_trace()
        x3 = x2 + x3
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
        