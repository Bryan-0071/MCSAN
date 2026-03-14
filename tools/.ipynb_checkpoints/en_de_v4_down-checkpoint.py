import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ipdb import set_trace
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
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        self.output_filter = output_filter = 16
        kernel_size = 3
        stride = 1
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv_rgb = ConvLayer(input_nc, output_filter, 1, stride)
        self.conv_dem = ConvLayer(5, output_filter, 1, stride)
        
        self.EB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder 1_3:第一行第三列
        self.DB2_1 = block(nb_filter[0] + nb_filter[1], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[1] + nb_filter[2], nb_filter[2], kernel_size, 1)
        self.DB3_2 = block(nb_filter[2]*2 + nb_filter[1], nb_filter[2], kernel_size, 1)

        self.DB4_1 = block(nb_filter[2]+ nb_filter[3], nb_filter[3], kernel_size, 1)
        self.DB4_2 = block(nb_filter[2]+ nb_filter[3] * 2, nb_filter[3], kernel_size, 1)
        self.DB4_3 = block(nb_filter[2]+ nb_filter[3] * 3, nb_filter[3], kernel_size, 1)
        
        self.gap = GAP2D()
        
        self.fc = nn.Linear(nb_filter[3], 2)
        
        self.fc_l = nn.Linear(output_filter * 128 * 128, 2)
        '''
        self.conv_fc = nn.Sequential(
            nn.Conv2d(output_filter, output_filter // 4, kernel_size=1),  # 减少通道数
            nn.ReLU(),
            nn.Conv2d(output_filter // 4, output_filter // 16, kernel_size=1),  # 进一步减少通道数
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 将特征图大小降为1x1
        )
        '''
        '''
        self.fc = nn.Sequential(
            nn.Conv2d(output_filter, 128, kernel_size=1),  # 瓶颈层，减少特征图的深度
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=1),  # 进一步减少深度到类别数
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),  # 全局平均池化，将空间维度减少到1x1
        )
        '''
        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_filter, 1, stride)

    def encoder(self, input):
        # print(input.shape)
        if input.shape[1] == 3:
            x = self.conv_rgb(input)
        else:
            x = self.conv_dem(input)
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

        return [x1_0, x2_0, x3_0, x4_0]

    def decoder(self, f_en):
        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))
        x1_1 = self.DB2_1(torch.cat([self.pool(f_en[0]), f_en[1]], 1))
        x2_1 = self.DB3_1(torch.cat([self.pool(f_en[1]), f_en[2]], 1))
        x2_2 = self.DB3_2(torch.cat([f_en[2], self.pool(x1_1), x2_1], 1))
        
        x3_1 = self.DB4_1(torch.cat([f_en[3], self.pool(f_en[2])], 1))
        x3_2 = self.DB4_2(torch.cat([f_en[3], x3_1, self.pool(x2_1)], 1))
        x3_3 = self.DB4_3(torch.cat([f_en[3], x3_1, x3_2, self.pool(x2_2)], 1))
        # set_trace()
        # output1 = self.conv_out(x1_3)
        # 这里的维度应该会有问题
        # print(x1_3.shape)
        output1 = self.gap(x3_3)
        # out = self.fc_l(x1_3.view(-1, self.output_filter * 128 * 128))
        out = self.fc(output1)
        
        # out = self.fc(output1.view(-1, self.output_filter * 128 * 128))
        return out