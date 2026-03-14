import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py
import cv2
import random
import os
# landslide dataset for sense4
# modify from https://github.com/iarai/Landslide4Sense-2022/blob/main/dataset/landslide_dataset.py

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, max_iters=None, augment=False):
        self.list_path = list_path
        self.data_dir = data_dir
        self.augment = augment
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        # 统一处理：读取图像和对应的 0/1 分类标签（label已在image文件中）
        for name in self.img_ids:
            img_file = data_dir + name
            self.files.append({
                'img': img_file,
                'name': name
            })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # 读取图像和标签（label已在image文件中）
        with h5py.File(datafiles['img'], 'r') as hf:
            image = hf['img'][:]
            # 直接从image文件中读取label
            if 'label' in hf:
                label = hf['label'][:]
            else:
                raise ValueError(f"未找到label字段在 {datafiles['img']}")
        
        name = datafiles['name']
        
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        
        # 图像转为 (C, H, W) 格式
        image = image.transpose((-1, 0, 1))
        # Data augmentation (applied on training set when augment=True)
        size = image.shape

        # 应用标准化
        for i in range(len(self.mean)):
            image[i, :, :] -= self.mean[i]
            image[i, :, :] /= self.std[i]
        # print(image[4, 0, 0], label, name)  # 打印第5个通道的第一个像素值、标签和文件名，检查数据是否正确加载和处理
        # print(image[10, 0, 0])  # 打印第11个通道的第一个像素值，检查数据是否正确加载和处理
        # 直接从数据中提取 RGB (B4,B3,B2 -> indices 3,2,1) 和 DEM (B14 -> index 13)
        # 返回 shape 为 (4, H, W)
        # selected = image[[0,1,2,3,4,5,6,7,8,9,10,11, 13], :, :].copy()
        selected = image[[3, 2, 1, 12,13], :, :].copy()
        
        size = selected.shape
        return selected, label.copy(), np.array(size), name

if __name__ == '__main__':
    
    train_dataset = LandslideDataSet(data_dir='F:/data/landslide/', list_path='F:/data/landslide/train.txt')
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(train_loader)
    for data, label, _, _ in train_loader:  # label 现在直接从image文件中读取
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data**2, dim=[0, 2, 3])

    mean = channels_sum / num_batches
    std = (channel_squared_sum / num_batches - mean**2)**0.5
    print(mean, std)
    #[-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
    #[0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]