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
from skimage.feature import greycomatrix, greycoprops
from torchvision import transforms
from configs.TFB_1 import args
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

def read_dataloader(path_bg, path_no_bg, mode):
    # 1. Prepare data
    features = []
    labels = []
    max_cu = 0
    min_cu = 100
    # data_list = []
    folder_list = path_bg + path_no_bg
    # 定义Transform
    if mode == 'train':
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            # MinMaxNormalize(),
            # transforms.Normalize(mean=[0.5]*8, std=[0.5]*8)
        ])

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
                # set_trace()
                
                # set_trace()
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)

                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    # if np.amax(curvature_value) > max_cu:
                    #     max_cu = np.amax(curvature_value)
                    # if np.amin(curvature_value) < min_cu:
                    #     min_cu = np.amin(curvature_value)
                    
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




                # Combine all features into a single vector and append to the feature list
                # feature = np.stack([re_image_data[:,:,0], re_image_data[:,:,1], re_image_data[:,:,2],slope, aspect, curvature_value,
                #                     hillshade_value, canny_edges], axis=-1)
                # if mode == 'train':
                #     feature_tensor = transform(feature)
                feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value,
                                    hillshade_value, canny_edges)
                features.append(feature_tensor)
                
            # Append the label to the label list
                labels.append(1)

        elif folder in path_no_bg:
            print(path_no_bg)
            # 随机选择 1000 个文件
            num = num_samples[cnt_no]
            cnt_no += 1
            num = min(num, len(dem_files))
            random_indices = random.sample(range(len(dem_files)), num)

            # 获取随机选择的文件列表
            selected_dem_files = [dem_files[i] for i in random_indices]
            selected_image_files = [image_files[i] for i in random_indices]
            print("len(selected_dem_files) = ",len(selected_dem_files))
            for dem_file, image_file in zip(selected_dem_files, selected_image_files):
                # print(dem_file,image_file)
                # Open DEM file
                with rasterio.open(dem_file) as ds:
                    Z = ds.read(1)
                    # Calculate slope, aspect, curvature, hillshade and resize them
                    slope =cv2.resize( calc_slope(Z, *ds.transform[0:2]), feature_size)
                    aspect =cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
                    curvature_value = cv2.resize(curvature(Z), feature_size)
                    # if np.amax(curvature_value) > max_cu:
                    #     max_cu = np.amax(curvature_value)
                    # if np.amin(curvature_value) < min_cu:
                    #     min_cu = np.amin(curvature_value)
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
                    # Combine all features into a single vector and append to the feature list
                # feature_2 = np.stack([re_image_data[:,:,0], re_image_data[:,:,1], re_image_data[:,:,2] ,slope, aspect, curvature_value,
                #                     hillshade_value, 
                #                     canny_edges], axis=-1)
                # if mode == 'train':
                #     feature_tensor_2 = transform(feature_2)
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



from concurrent.futures import ThreadPoolExecutor

def process_files(dem_file, image_file, feature_size):
    with rasterio.open(dem_file) as ds:
        Z = ds.read(1)
        slope = cv2.resize(calc_slope(Z, *ds.transform[0:2]), feature_size)
        aspect = cv2.resize(calc_aspect(Z, *ds.transform[0:2]), feature_size)
        curvature_value = cv2.resize(curvature(Z), feature_size)
        hillshade_value = cv2.resize(hillshade(Z, 315, 45), feature_size)
        
    with rasterio.open(image_file) as ds:
        img = ds.read()
        image_data = img[:3, :, :].transpose((1, 2, 0)).astype(np.uint8)
        re_image_data = cv2.resize(image_data, feature_size)
        canny_edges = cv2.resize(cv2.Canny(image_data, threshold1=200, threshold2=400), feature_size)

    feature_tensor = Normalization(re_image_data, slope, aspect, curvature_value, hillshade_value, canny_edges)
    return feature_tensor

def read_dataloader_fast(path_bg, path_no_bg, mode):
    features = []
    labels = []
    folder_list = path_bg + path_no_bg

    feature_size = args.feature_size

    with ThreadPoolExecutor() as executor:
        futures = []
        for folder in folder_list:
            dem_files = sorted(glob.glob(os.path.join(folder, 'dsm', '*.tif')))
            image_files = sorted(glob.glob(os.path.join(folder, '遥感', '*.tif')))
            for dem_file, image_file in zip(dem_files, image_files):
                futures.append(executor.submit(process_files, dem_file, image_file, feature_size))
                if folder in path_bg:
                    labels.append(1)
                else:
                    labels.append(0)

        for future in futures:
            features.append(future.result())

    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(features_tensor, labels_tensor)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if mode == 'test':
        for batch in dataloader:
            feature_batch, label_batch = batch
            print(feature_batch.shape, label_batch.shape)

    return dataloader

def save_model(model, path):
    folder_name = '/home/lab1015/programmes/guang/code/beng/save_model/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_name)

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
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
        recall = recall_score(y_true, y_pred, average='binary')  # 二分类问题
        f1 = f1_score(y_true, y_pred, average='binary')  # 二分类问题
        print(f"Test Accuracy: {100 * correct / total:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        new_acc = 100 * correct / total
        if new_acc > best_acc:
            best_acc = new_acc
            save_model(model=model,path=path)
    return best_acc