import os
import glob
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import rasterio
import cv2
import matplotlib.pyplot as plt
from numpy import gradient, pi, sin, cos, arctan2, sqrt
from scipy.ndimage import generic_filter
import matplotlib
from torchvision import transforms
from configs.landslide_sense4 import args
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, f1_score
from ipdb import set_trace
from torch.cuda.amp import GradScaler, autocast
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import json
# 设置随机数种子以确保可重复性
random_seed = 42
random.seed(random_seed)
# Function for calculating NDVI from NIR and Red bands
def calculate_ndvi(nir_band, red_band):
    return (nir_band - red_band) / (nir_band + red_band)

def calc_slope(Z, x, y): # 范围是 [0, π/2]
    Gx, Gy = gradient(Z)
    G = sqrt(Gx**2 + Gy**2)
    slope = np.arctan(G)
    return slope

def calc_aspect(Z, x, y): # 范围是 [-π, π]
    Gx, Gy = gradient(Z)
    aspect = arctan2(-Gy, Gx)
    return aspect

# https://blog.csdn.net/weixin_43593330/article/details/107543737
def curvature(Z): # 曲率train 137.63235473632812,-82.44840240478516 曲率150.32962036132812,-88.77801513671875
    laplacian = generic_filter(Z, np.mean, size=3, mode='constant')
    curvature = Z - laplacian
    return curvature

def hillshade(array, azimuth, angle_altitude): # 范围调整到 [0, 255]
    azimuthrad = azimuth*pi / 180
    altituderad = angle_altitude*pi / 180

    x, y = gradient(array)
    slope = pi/2. - np.arctan(sqrt(x*x + y*y))
    aspect = arctan2(-x, y)

    shaded = sin(altituderad) * sin(slope) + cos(altituderad) * cos(slope) * cos(-azimuthrad - aspect - pi / 2.)

    return 255*(shaded + 1)/2

def image_enhancement(image):
    """Image enhancement using histogram equalization"""
    # Scale the image pixel values to range from 0 to 255
    img_scaled = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
    # Convert the scaled image to uint8 data type
    img_uint8 = img_scaled.astype(np.uint8)
    # Now you can apply histogram equalization
    return cv2.equalizeHist(img_uint8)

def binarization(image, threshold=128):
    """Binarization of an image using a threshold"""
    _, binarized = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binarized

def edge_detection(image):
    """Edge detection using the Canny operator"""
    return cv2.Canny(image, 100, 200)

def calc_glcm_features(image, distances, angles):
    image_uint8 = (image * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    glcm = greycomatrix(image_uint8, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    return np.concatenate([contrast.flatten(), dissimilarity.flatten(),
                           homogeneity.flatten(), energy.flatten(),
                           correlation.flatten()])

def read_data(path_bg, path_no_bg):
    # 1. Prepare data
    features = []
    labels = []

    # Save the features of the specified image for visualization
    visualize_features = None
    rgb_img_list = []
    dem_img_list = []
    input_tensor_list = []
    feature_size = args.feature_size
    for folder in [path_bg, path_no_bg]:
        dem_files = sorted(glob.glob(os.path.join(folder, 'dsm', '*.tif')))
        image_files = sorted(glob.glob(os.path.join(folder, '遥感', '*.tif')))
        if folder == path_bg:
            for dem_file, image_file in zip(dem_files, image_files):
                # print(dem_file,image_file)
                #set_trace()
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)

                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
                    if len(dem_img_list) < 16:
                        tmp = hillshade_value
                        tmp = np.expand_dims(tmp, axis=-1)
                        tmp = np.float32(tmp)
                        dem_img_list.append(tmp)
                        # set_trace()
                # Open image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                    if len(rgb_img_list) < 16:
                        tmp = re_image_data
                        tmp = np.float32(tmp)
                        rgb_img_list.append(tmp)
                        
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=200, threshold2=400), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

            # Combine all features into a single vector and append to the feature list
                feature = np.stack([re_image_data[:,:,0], re_image_data[:,:,1], re_image_data[:,:,2],slope, aspect, curvature_value,
                                    hillshade_value, 
                                    canny_edges], axis=-1)
                if len(input_tensor_list) < 16:
                    # input_tensor = preprocess_image(feature, mean=[0.5, 0.5, 0.5],
                    #                                     std=[0.5, 0.5, 0.5])
                    input_tensor = feature
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    input_tensor = torch.from_numpy(input_tensor).float()
                    input_tensor = input_tensor.cuda()

                    input_tensor = input_tensor.permute(0, 3, 1, 2)
                    input_tensor_list.append(input_tensor)
                features.append(feature)
                
            # Append the label to the label list
                labels.append(1)

        if folder == path_no_bg:
            for dem_file, image_file in zip(dem_files, image_files):
                # print(dem_file,image_file)
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope =cv2.resize( calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect =cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value =cv2.resize(hillshade(Z, 315, 45), feature_size)
                    # set_trace()
                    if len(dem_img_list) < 32:
                        tmp = hillshade_value
                        tmp = np.expand_dims(tmp, axis=-1)
                        tmp = np.float32(tmp)
                        dem_img_list.append(tmp)
                    # Process each image file
                with rasterio.open(image_file) as ds:
                        # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                    if len(rgb_img_list) < 32:
                        tmp = re_image_data
                        tmp = np.float32(tmp)
                        rgb_img_list.append(tmp)
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""
                    # Combine all features into a single vector and append to the feature list
                feature = np.stack([re_image_data[:,:,0], re_image_data[:,:,1], re_image_data[:,:,2] ,slope, aspect, curvature_value,
                                    hillshade_value, 
                                    canny_edges], axis=-1)
                if len(input_tensor_list) < 32:
                    # input_tensor = preprocess_image(feature, mean=[0.5, 0.5, 0.5],
                    #                                     std=[0.5, 0.5, 0.5])
                    input_tensor_2 = feature
                    input_tensor_2 = np.expand_dims(input_tensor_2, axis=0)
                    input_tensor_2 = torch.from_numpy(input_tensor_2).float()
                    input_tensor_2 = input_tensor_2.cuda()

                    input_tensor_2 = input_tensor_2.permute(0, 3, 1, 2)
                    input_tensor_list.append(input_tensor_2)
                features.append(feature)

                    # Append the label to the label list
                labels.append(0)        # If this is the specified image, save the features for visualization
    return features, labels

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)


def Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, 
                                    canny_edges):
    # slope 范围是 [0, π/2]
    # aspect 范围是 [-π, π]
    # curvature_value 范围是(-200,200)
    # hillshade_value 范围调整到 [0, 255]
    # canny_edges 范围是 [0, 255]
    # 所有结果都归一化到[-1, 1]

    d_t_re_image_data = torch.from_numpy(re_image_data.transpose((2,0,1)))
    d_t_re_image_data = d_t_re_image_data.float().div(255)
    d_t_re_image_data = d_t_re_image_data.sub_(0.5).div_(0.5)

    # set_trace()
    d_t_slope = torch.from_numpy(slope).unsqueeze(0)
    d_t_slope = d_t_slope.float().div(math.pi/2)
    d_t_slope= d_t_slope.sub_(0.5).div_(0.5)

    d_t_aspect = torch.from_numpy(aspect).unsqueeze(0)
    d_t_aspect = d_t_aspect.float().div(math.pi)

    d_t_curvature_value = torch.from_numpy(curvature_value).unsqueeze(0)
    d_t_curvature_value = torch.clamp(d_t_curvature_value, min=-200, max=200)
    d_t_curvature_value = d_t_curvature_value.float().div(200)

    d_t_hillshade_value = torch.from_numpy(hillshade_value).unsqueeze(0)
    d_t_hillshade_value = d_t_hillshade_value.float().div(255)
    d_t_hillshade_value = d_t_hillshade_value.sub_(0.5).div_(0.5)

    d_t_canny_edges = torch.from_numpy(canny_edges).unsqueeze(0)
    d_t_canny_edges = d_t_canny_edges.float().div(255)
    d_t_canny_edges = d_t_canny_edges.sub_(0.5).div_(0.5)

    # 按照通道堆叠
    d_t_stacked = torch.cat((d_t_re_image_data, d_t_slope, d_t_aspect, d_t_curvature_value, d_t_hillshade_value, d_t_canny_edges), dim=0)
    # set_trace()
    return d_t_stacked

def Normalization_5(slope, aspect, curvature_value,
                                    hillshade_value, 
                                    canny_edges):
    # slope 范围是 [0, π/2]
    # aspect 范围是 [-π, π]
    # curvature_value 范围是(-200,200)
    # hillshade_value 范围调整到 [0, 255]
    # canny_edges 范围是 [0, 255]
    # 所有结果都归一化到[-1, 1]

    # set_trace()
    d_t_slope = torch.from_numpy(slope).unsqueeze(0)
    d_t_slope = d_t_slope.float().div(math.pi/2)
    d_t_slope= d_t_slope.sub_(0.5).div_(0.5)

    d_t_aspect = torch.from_numpy(aspect).unsqueeze(0)
    d_t_aspect = d_t_aspect.float().div(math.pi)

    d_t_curvature_value = torch.from_numpy(curvature_value).unsqueeze(0)
    d_t_curvature_value = torch.clamp(d_t_curvature_value, min=-200, max=200)
    d_t_curvature_value = d_t_curvature_value.float().div(200)

    d_t_hillshade_value = torch.from_numpy(hillshade_value).unsqueeze(0)
    d_t_hillshade_value = d_t_hillshade_value.float().div(255)
    d_t_hillshade_value = d_t_hillshade_value.sub_(0.5).div_(0.5)

    d_t_canny_edges = torch.from_numpy(canny_edges).unsqueeze(0)
    d_t_canny_edges = d_t_canny_edges.float().div(255)
    d_t_canny_edges = d_t_canny_edges.sub_(0.5).div_(0.5)

    # 按照通道堆叠
    d_t_stacked = torch.cat(( d_t_slope, d_t_aspect, d_t_curvature_value, d_t_hillshade_value, d_t_canny_edges), dim=0)
    # set_trace()
    return d_t_stacked

def read_dataloader(path_bg, path_no_bg, mode):
    # 1. Prepare data
    features = []
    labels = []
    max_cu = 0
    min_cu = 100
    # data_list = []
    folder_list = path_bg + path_no_bg
    # 定义Transform

    feature_size = args.feature_size
    num_samples = []
    cnt_no = 0
    for folder in folder_list:
        # set_trace()
        dem_files = sorted(glob.glob(os.path.join(folder, 'dsm', '*.tif')))
        image_files = sorted(glob.glob(os.path.join(folder, '遥感', '*.tif')))
        if folder in path_bg:
            num_samples.append(len(dem_files) * 3)
            print(folder)
            for dem_file, image_file in zip(dem_files, image_files):
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)

                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
                    
                # Open image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                        
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

                feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor)
                
            # Append the label to the label list
                labels.append(1)

        elif folder in path_no_bg:
            print(folder)

            # 取样数量 为 len(dem_files) * 3 与 非崩岗数量的较小值
            num = num_samples[cnt_no]
            cnt_no += 1
            num = min(num, len(dem_files))
            random_indices = random.sample(range(len(dem_files)), num)
            # print("random_indices = ",random_indices)
            # 获取随机选择的文件列表
            selected_dem_files = [dem_files[i] for i in random_indices]
            selected_image_files = [image_files[i] for i in random_indices]
            # selected_dem_files = dem_files
            # selected_image_files = image_files
            print("len(selected_dem_files) = ",len(selected_dem_files))
            for dem_file, image_file in zip(selected_dem_files, selected_image_files):
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope =cv2.resize( calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect =cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)

                    hillshade_value =cv2.resize(hillshade(Z, 315, 45), feature_size)

                    # Process each image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""
                feature_tensor_2 = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor_2)

                # Append the label to the label list
                labels.append(0)        # If this is the specified image, save the features for visualization
    # 将features和labels转换为PyTorch张量
    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)

    # 创建TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    # 创建DataLoader
    batch_size = args.batch_size  # 根据需要调整批量大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # 测试DataLoader
    # if mode == 'test':
    #     for batch in dataloader:
    #         feature_batch, label_batch = batch
    #         print(feature_batch.shape, label_batch.shape)
    print("曲率================================{0},{1}".format(max_cu,min_cu))
    return dataloader


def save_model(model, path):
    path = str(path)
    folder_name = '/home/lab1015/programmes/guang/code/beng/save_model/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_name)
        print(folder_name)

    # 保存模型的状态字典
    torch.save(model.state_dict(), folder_name+"/1.pth")

def load_model(model, path):
    folder_name = '/home/lab1015/programmes/guang/code/beng/save_model/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        print("error没有该模型")
    # 实例化模型
    # model = model_class()
    model_path = folder_name+"/1.pth"
    # 加载模型状态字典
    model.load_state_dict(torch.load(model_path))
    return model
def test_model_in_epoch(model, test_loader, path, best_acc):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for images, labels in test_loader:
            # images = (images - mean) / std
            images = images.cuda()
            labels = labels.cuda()
            # input1 = torch.clone(images[:,:3,:,:])
            # input2 = torch.clone(images[:,3:,:,:])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        recall = recall_score(y_true, y_pred, average='macro')  # 二分类问题
        f1 = f1_score(y_true, y_pred, average='macro')  # 二分类问题
        print(f"Test Accuracy: {100 * correct / total:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        new_acc = 100 * correct / total
        if new_acc > best_acc:
            best_acc = new_acc
            if path is not None:
                save_model(model=model,path=path)
    return best_acc

def test_model_in_epoch_sense4(model, test_loader, path, best_acc):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for batch_id, src_data in enumerate(test_loader):
            # images = (images - mean) / std
            images, labels, _, _ = src_data
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.squeeze(1)  # 压缩标签为 (batch_size,) 形状
            labels = labels.long()  # 确保标签是整数类型 (LongTensor)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        recall = recall_score(y_true, y_pred, average='macro')  # 二分类问题
        f1 = f1_score(y_true, y_pred, average='macro')  # 二分类问题
        print(f"Valid Accuracy: {100 * correct / total:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        new_acc = 100 * correct / total
        if new_acc > best_acc:
            best_acc = new_acc
            if path is not None:
                save_model(model=model,path=path)
    return best_acc

def test_model_in_epoch_single(model, test_loader, path, best_acc,in_a = 0,
in_b = 3,
in_c = 3,
in_d = 4):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for images, labels in test_loader:
            # images = (images - mean) / std
            images = images.cuda()
            labels = labels.cuda()
            # input1 = torch.clone(images[:,in_a:in_b,:,:])
            # input2 = torch.clone(images[:,in_c:,:,:])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
        recall = recall_score(y_true, y_pred, average='macro')  # 二分类问题
        f1 = f1_score(y_true, y_pred, average='macro')  # 二分类问题
        print(f"Test Accuracy: {100 * correct / total:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        new_acc = 100 * correct / total
        if new_acc > best_acc:
            best_acc = new_acc
            if path and best_acc>90:
                save_model(model=model,path=path)
    return best_acc

import re
def sort_naturally(file_list):
    """Sort a list of filenames naturally by numbers in filenames and return the sorted filenames and their indices."""
    def natural_key(filename):
        parts = re.split('([0-9]+)', filename)
        return [int(part) if part.isdigit() else part for part in parts]
    
    sorted_files = sorted(file_list, key=natural_key)
    indices = [file_list.index(file) for file in sorted_files]
    
    return sorted_files, indices

# 提取数字的函数
def extract_number(filename):
    
    match = re.search(r'region_dsm_(\d+)\.tif', filename)

    if match:
        number = match.group(1)
        # print(number)  # 输出 1018
        return int(number)
    else:
        print("No number found in the file name.")
    

def read_dataloader_bg4(path_bg, path_no_bg, mode):
    # 1. Prepare data
    features = []
    labels = []
    max_cu = 0
    min_cu = 100
    # data_list = []
    folder_list = path_bg + path_no_bg
    # 定义Transform

    feature_size = args.feature_size
    num_samples = []
    numbers = []
    cnt_no = 0
    for folder in folder_list:
        # set_trace()
        dem_files = sorted(glob.glob(os.path.join(folder, 'dsm', '*.tif')))
        image_files = sorted(glob.glob(os.path.join(folder, '遥感', '*.tif')))
        # for filename in dem_files:
        #     extract_number(filename)
        tmp = [extract_number(file) for file in dem_files]
        numbers.extend(tmp)

        if folder in path_bg:
            print(folder)
            for dem_file, image_file in zip(dem_files, image_files):
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)

                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
                    
                # Open image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                        
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=200, threshold2=400), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

                feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor)
                
            # Append the label to the label list
                labels.append(1)

        elif folder in path_no_bg:
            print(path_no_bg)
            num = len(dem_files)
            print("len(selected_dem_files) = ",num)
            for dem_file, image_file in zip(dem_files, image_files):
                # print(dem_file,image_file)
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope =cv2.resize( calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect =cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value =cv2.resize(hillshade(Z, 315, 45), feature_size)

                    # Process each image file
                with rasterio.open(image_file) as ds:
                        # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

                feature_tensor_2 = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor_2)

                # Append the label to the label list
                labels.append(0)        # If this is the specified image, save the features for visualization
    # 将features和labels转换为PyTorch张量
    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)
    print(labels_tensor)
    print(numbers)
    # 将列表保存为JSON文件
    file_path = '/home/lab1015/programmes/guang/code/beng/save_model/json/numbers.json'  # 你想要保存的文件路径
    with open(file_path, 'w') as file:
        json.dump(numbers, file)
    # 创建TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    # 创建DataLoader
    batch_size = args.batch_size  # 根据需要调整批量大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    # 测试DataLoader
    # if mode == 'test':
    #     for batch in dataloader:
    #         feature_batch, label_batch = batch
    #         print(feature_batch.shape, label_batch.shape)
    print("曲率================================{0},{1}".format(max_cu,min_cu))
    return dataloader

def read_dataloader_out_order(path_bg, path_no_bg, path_bg_test, path_no_bg_test, mode):
    # 1. Prepare data
    features = []
    labels = []
    max_cu = 0
    min_cu = 100
    # data_list = []
    # folder_list = path_bg + path_bg_test + path_no_bg + path_no_bg_test
    folder_list = path_bg_test + path_no_bg_test
    # 定义Transform

    feature_size = args.feature_size
    num_samples = []
    cnt_no = 0
    for folder in folder_list:
        # set_trace()
        dem_files = sorted(glob.glob(os.path.join(folder, 'dsm', '*.tif')))
        image_files = sorted(glob.glob(os.path.join(folder, '遥感', '*.tif')))
        if (folder in path_bg) or (folder in path_bg_test):
            num_samples.append(len(dem_files) * 3)
            print(folder)
            for dem_file, image_file in zip(dem_files, image_files):
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)

                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
                    
                # Open image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                        
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

                feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor)
                
            # Append the label to the label list
                labels.append(1)

        elif (folder in path_no_bg) or (folder in path_no_bg_test):
            print(folder)

            # 取样数量 为 len(dem_files) * 3 与 非崩岗数量的较小值
            num = num_samples[cnt_no]
            cnt_no += 1
            num = min(num, len(dem_files))
            random_indices = random.sample(range(len(dem_files)), num)
            print("random_indices = ",random_indices)
            # 获取随机选择的文件列表
            selected_dem_files = [dem_files[i] for i in random_indices]
            selected_image_files = [image_files[i] for i in random_indices]
            # selected_dem_files = dem_files
            # selected_image_files = image_files
            print("len(selected_dem_files) = ",len(selected_dem_files))
            for dem_file, image_file in zip(selected_dem_files, selected_image_files):
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope =cv2.resize( calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect =cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)

                    hillshade_value =cv2.resize(hillshade(Z, 315, 45), feature_size)

                    # Process each image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""
                feature_tensor_2 = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor_2)

                # Append the label to the label list
                labels.append(0)        # If this is the specified image, save the features for visualization
    # 将features和labels转换为PyTorch张量
    # features_tensor = torch.stack(features)
    # labels_tensor = torch.tensor(labels)

    # # 创建TensorDataset
    # dataset = TensorDataset(features_tensor, labels_tensor)

    # 创建DataLoader
    batch_size = args.batch_size  # 根据需要调整批量大小
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    features=np.array(features)
    labels=np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=40)
    dX_train=X_train;dX_test=X_test;dy_train = y_train;dy_test =y_test
    dX_train= dX_train.astype('float32')
    dX_test = dX_test.astype('float32')

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(dX_train).float()
    y_train_tensor = torch.tensor(dy_train).long()
    X_test_tensor = torch.tensor(dX_test).float()
    y_test_tensor = torch.tensor(dy_test).long()
    # X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
    # X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 测试DataLoader
    # if mode == 'test':
    #     for batch in dataloader:
    #         feature_batch, label_batch = batch
    #         print(feature_batch.shape, label_batch.shape)
    print("曲率================================{0},{1}".format(max_cu,min_cu))
    return train_loader,test_loader

def read_dataloader_aug(path_bg, path_no_bg, mode):
    # 1. Prepare data
    features = []
    labels = []
    max_cu = 0
    min_cu = 100
    # data_list = []
    folder_list = path_bg + path_no_bg
    # 定义Transform

    feature_size = args.feature_size
    num_samples = []
    cnt_no = 0
    for folder in folder_list:
        # set_trace()
        dem_files = sorted(glob.glob(os.path.join(folder, 'dsm', '*.tif')))
        image_files = sorted(glob.glob(os.path.join(folder, '遥感', '*.tif')))
        image_files_aug = sorted(glob.glob(os.path.join(folder, 'rgb_aug', '*.tif')))
        dem_files_aug = sorted(glob.glob(os.path.join(folder, 'dsm_aug', '*.tif')))

        if folder in path_bg:
            num_samples.append(len(dem_files) * 3)
            print(folder)
            for dem_file, image_file in zip(dem_files, image_files):
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
                    
                # Open image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                        
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

                feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor)
                
            # Append the label to the label list
                labels.append(1)

            for dem_file, image_file in zip(dem_files_aug, image_files_aug):
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
                    
                # Open image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                        
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""

                feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor)
                
            # Append the label to the label list
                labels.append(1)

            print(len(features))

        elif folder in path_no_bg:
            print(folder)

            # 取样数量 为 len(dem_files) * 3 与 非崩岗数量的较小值
            num = num_samples[cnt_no]
            cnt_no += 1
            num = min(num, len(dem_files))
            random_indices = random.sample(range(len(dem_files)), num)
            # print("random_indices = ",random_indices)
            # 获取随机选择的文件列表
            selected_dem_files = [dem_files[i] for i in random_indices]
            selected_image_files = [image_files[i] for i in random_indices]
            # selected_dem_files = dem_files
            # selected_image_files = image_files
            print("len(selected_dem_files) = ",len(selected_dem_files))
            for dem_file, image_file in zip(selected_dem_files, selected_image_files):
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope =cv2.resize( calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect =cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)

                    hillshade_value =cv2.resize(hillshade(Z, 315, 45), feature_size)

                    # Process each image file
                with rasterio.open(image_file) as ds:
                    # Assuming the NIR band is the fourth band and Red band is the third band
                    img = ds.read()
                    image_data = img[:3, :, :].transpose((1, 2, 0))
                    # Convert image_data to uint8
                    image_data = image_data.astype(np.uint8)
                    re_image_data = cv2.resize(image_data, feature_size)
                    # Apply Canny high-pass filter to image_data
                    canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=100, threshold2=200), feature_size)
                    # Apply high-pass filter to the image
                    """hpf_image = high_pass_filter(image_data)"""
                # feature_tensor_2 = Normalization(re_image_data, slope, aspect, curvature_value,
                #                     hillshade_value, canny_edges)
                feature_tensor_2 = Normalization(re_image_data, slope, aspect, curvature_value,
                                   hillshade_value, canny_edges)
                features.append(feature_tensor_2)

                # Append the label to the label list
                labels.append(0)        # If this is the specified image, save the features for visualization
    # 将features和labels转换为PyTorch张量
    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)

    # 创建TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    # 创建DataLoader
    batch_size = args.batch_size  # 根据需要调整批量大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("曲率================================{0},{1}".format(max_cu,min_cu))
    return dataloader