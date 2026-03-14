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

def print_cam(model, path, rgb_img_list, dem_img_list, input_tensor_list):
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # 文件夹名称
    folder_name = '/home/lab1015/programmes/guang/code/beng/img/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_name)
    targets = None
    target_layers_fusion = [model.funsion.fusion_block4.conv2.conv2d]
   # target_layers_fusion = [model.fu1.residual.conv3]
    target_layers_rgb_4 = [model.EB4_0[1][0].conv]
    target_layers_dem_4 = [model.EB4_dem[1][0].conv]

    target_layers_rgb_3 = [model.EB3_0[1][0].conv]
    target_layers_dem_3 = [model.EB3_dem[1][0].conv]

    target_layers_rgb_2 = [model.EB2_0[1][0].conv]
    target_layers_dem_2 = [model.EB2_dem[1][0].conv]

    target_layers_rgb_1 = [model.EB1_0[0].conv]
    target_layers_dem_1 = [model.EB1_dem[0].conv]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam_fusion = GradCAM(model=model,
                                target_layers=target_layers_fusion,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_4 = GradCAM(model=model,
                                target_layers=target_layers_rgb_4,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_4 = GradCAM(model=model,
                                target_layers=target_layers_dem_4,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_3 = GradCAM(model=model,
                                target_layers=target_layers_rgb_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_3 = GradCAM(model=model,
                                target_layers=target_layers_dem_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_2 = GradCAM(model=model,
                                target_layers=target_layers_rgb_2,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_2 = GradCAM(model=model,
                                target_layers=target_layers_dem_2,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_1 = GradCAM(model=model,
                                target_layers=target_layers_rgb_1,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_1 = GradCAM(model=model,
                                target_layers=target_layers_dem_1,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    # cam.batch_size = 32

    for i in range(len(rgb_img_list)):

        grayscale_cam_fu = cam_fusion(input_tensor=input_tensor_list[i],
                            targets=targets,
                            #eigen_smooth=args.eigen_smooth,
                            # aug_smooth=args.aug_smooth
                            )
        
        grayscale_cam_rgb_4 = cam_rgb_4(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_4 = cam_dem_4(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        
        grayscale_cam_rgb_3 = cam_rgb_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_3 = cam_dem_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )

        grayscale_cam_rgb_2 = cam_rgb_2(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_2 = cam_dem_2(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )

        grayscale_cam_rgb_1 = cam_rgb_1(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_1 = cam_dem_1(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        
        grayscale_cam_fu = grayscale_cam_fu[0, :]
        
        grayscale_cam_rgb_4 = grayscale_cam_rgb_4[0, :]
        grayscale_cam_dem_4 = grayscale_cam_dem_4[0, :]
        
        grayscale_cam_rgb_3 = grayscale_cam_rgb_3[0, :]
        grayscale_cam_dem_3 = grayscale_cam_dem_3[0, :]
        grayscale_cam_rgb_2 = grayscale_cam_rgb_2[0, :]
        grayscale_cam_dem_2 = grayscale_cam_dem_2[0, :]
        grayscale_cam_rgb_1 = grayscale_cam_rgb_1[0, :]
        grayscale_cam_dem_1 = grayscale_cam_dem_1[0, :]

        rgb_img = rgb_img_list[i]
        dem_img = dem_img_list[i]
        
        path_1 = path
        cv2.imwrite(f'./img/{path_1}/{i}_rgb.jpg', rgb_img)
        cv2.imwrite(f'./img/{path_1}/{i}_dem.jpg', dem_img)
        
        rgb_img = rgb_img / 255.0
        dem_img = dem_img / 255.0
        
        cam_image_fu = show_cam_on_image(rgb_img, grayscale_cam_fu)
        
        cam_image_rgb_4 = show_cam_on_image(rgb_img, grayscale_cam_rgb_4)
        cam_image_dem_4 = show_cam_on_image(rgb_img, grayscale_cam_dem_4)
        
        cam_image_rgb_3 = show_cam_on_image(rgb_img, grayscale_cam_rgb_3)
        cam_image_dem_3 = show_cam_on_image(rgb_img, grayscale_cam_dem_3)
        cam_image_rgb_2 = show_cam_on_image(rgb_img, grayscale_cam_rgb_2)
        cam_image_dem_2 = show_cam_on_image(rgb_img, grayscale_cam_dem_2)
        cam_image_rgb_1 = show_cam_on_image(rgb_img, grayscale_cam_rgb_1)
        cam_image_dem_1 = show_cam_on_image(rgb_img, grayscale_cam_dem_1)
        cam_image_dem_2_on_dem = show_cam_on_image(dem_img, grayscale_cam_dem_2)
        cam_image_dem_1_on_dem = show_cam_on_image(dem_img, grayscale_cam_dem_1)

        cv2.imwrite(f'./img/{path_1}/{i}_fusion.jpg', cam_image_fu)
        
        cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_4.jpg', cam_image_rgb_4)
        cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_4.jpg', cam_image_dem_4)
        
        cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_3.jpg', cam_image_rgb_3)
        cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_3.jpg', cam_image_dem_3)
        cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_2.jpg', cam_image_rgb_2)
        cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_2.jpg', cam_image_dem_2)
        cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_1.jpg', cam_image_rgb_1)
        cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_1.jpg', cam_image_dem_1)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_2_on_dem.jpg', cam_image_dem_2_on_dem)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_1_on_dem.jpg', cam_image_dem_1_on_dem)


def print_cam_resnest_hff(model, path, rgb_img_list, dem_img_list, input_tensor_list):
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # 文件夹名称
    folder_name = '/home/lab1015/programmes/guang/code/beng/img/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_name)
    targets = None
    target_layers_fusion = [model.fu4.residual.conv3]
   # target_layers_fusion = [model.fu1.residual.conv3]
    target_layers_rgb_4 = [model.layer4[2].conv3]
    target_layers_dem_4 = [model.layer_dsm4[2].conv3]

    target_layers_rgb_3 = [model.layer3[2].conv3]
    target_layers_dem_3 = [model.layer_dsm3[2].conv3]

    target_layers_rgb_2 = [model.layer2[2].conv3]
    target_layers_dem_2 = [model.layer_dsm2[2].conv3]

    # target_layers_rgb_1 = [model.EB1_0[0].conv]
    # target_layers_dem_1 = [model.EB1_dem[0].conv]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam_fusion = GradCAM(model=model,
                                target_layers=target_layers_fusion,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_4 = GradCAM(model=model,
                                target_layers=target_layers_rgb_4,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_4 = GradCAM(model=model,
                                target_layers=target_layers_dem_4,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_3 = GradCAM(model=model,
                                target_layers=target_layers_rgb_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_3 = GradCAM(model=model,
                                target_layers=target_layers_dem_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_2 = GradCAM(model=model,
                                target_layers=target_layers_rgb_2,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_2 = GradCAM(model=model,
                                target_layers=target_layers_dem_2,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    # cam_rgb_1 = GradCAM(model=model,
    #                             target_layers=target_layers_rgb_1,
    #                             # use_cuda=True,
    #                             # reshape_transform=reshape_transform
    #                             )
    # cam_dem_1 = GradCAM(model=model,
    #                             target_layers=target_layers_dem_1,
    #                             # use_cuda=True,
    #                             # reshape_transform=reshape_transform
    #                             )

    # cam.batch_size = 32

    for i in range(len(rgb_img_list)):

        grayscale_cam_fu = cam_fusion(input_tensor=input_tensor_list[i],
                            targets=targets,
                            #eigen_smooth=args.eigen_smooth,
                            # aug_smooth=args.aug_smooth
                            )
        
        grayscale_cam_rgb_4 = cam_rgb_4(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_4 = cam_dem_4(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        
        grayscale_cam_rgb_3 = cam_rgb_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_3 = cam_dem_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )

        # grayscale_cam_rgb_2 = cam_rgb_2(input_tensor=input_tensor_list[i],
        #                     targets=targets,
        #                     )
        # grayscale_cam_dem_2 = cam_dem_2(input_tensor=input_tensor_list[i],
        #                     targets=targets,
        #                     )

        # grayscale_cam_rgb_1 = cam_rgb_1(input_tensor=input_tensor_list[i],
        #                     targets=targets,
        #                     )
        # grayscale_cam_dem_1 = cam_dem_1(input_tensor=input_tensor_list[i],
        #                     targets=targets,
        #                     )
        
        grayscale_cam_fu = grayscale_cam_fu[0, :]
        
        grayscale_cam_rgb_4 = grayscale_cam_rgb_4[0, :]
        grayscale_cam_dem_4 = grayscale_cam_dem_4[0, :]
        
        grayscale_cam_rgb_3 = grayscale_cam_rgb_3[0, :]
        grayscale_cam_dem_3 = grayscale_cam_dem_3[0, :]
        # grayscale_cam_rgb_2 = grayscale_cam_rgb_2[0, :]
        # grayscale_cam_dem_2 = grayscale_cam_dem_2[0, :]
        # grayscale_cam_rgb_1 = grayscale_cam_rgb_1[0, :]
        # grayscale_cam_dem_1 = grayscale_cam_dem_1[0, :]

        rgb_img = rgb_img_list[i]
        # dem_img = dem_img_list[i]
        
        path_1 = path
        cv2.imwrite(f'./img/{path_1}/{i}_rgb.jpg', rgb_img)
        # cv2.imwrite(f'./img/{path_1}/{i}_dem.jpg', dem_img)
        
        rgb_img = rgb_img / 255.0
        # dem_img = dem_img / 255.0
        
        cam_image_fu = show_cam_on_image(rgb_img, grayscale_cam_fu)
        
        cam_image_rgb_4 = show_cam_on_image(rgb_img, grayscale_cam_rgb_4)
        cam_image_dem_4 = show_cam_on_image(rgb_img, grayscale_cam_dem_4)
        
        cam_image_rgb_3 = show_cam_on_image(rgb_img, grayscale_cam_rgb_3)
        cam_image_dem_3 = show_cam_on_image(rgb_img, grayscale_cam_dem_3)
        # cam_image_rgb_2 = show_cam_on_image(rgb_img, grayscale_cam_rgb_2)
        # cam_image_dem_2 = show_cam_on_image(rgb_img, grayscale_cam_dem_2)
        # cam_image_rgb_1 = show_cam_on_image(rgb_img, grayscale_cam_rgb_1)
        # cam_image_dem_1 = show_cam_on_image(rgb_img, grayscale_cam_dem_1)
        # cam_image_dem_2_on_dem = show_cam_on_image(dem_img, grayscale_cam_dem_2)
        # cam_image_dem_1_on_dem = show_cam_on_image(dem_img, grayscale_cam_dem_1)

        cv2.imwrite(f'./img/{path_1}/{i}_fusion.jpg', cam_image_fu)
        
        cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_4.jpg', cam_image_rgb_4)
        cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_4.jpg', cam_image_dem_4)
        
        cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_3.jpg', cam_image_rgb_3)
        cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_3.jpg', cam_image_dem_3)
        # cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_2.jpg', cam_image_rgb_2)
        # cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_2.jpg', cam_image_dem_2)
        # cv2.imwrite(f'./img/{path_1}/{i}_rgb_layer_1.jpg', cam_image_rgb_1)
        # cv2.imwrite(f'./img/{path_1}/{i}_dem_layer_1.jpg', cam_image_dem_1)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_2_on_dem.jpg', cam_image_dem_2_on_dem)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_1_on_dem.jpg', cam_image_dem_1_on_dem)




def print_cam_landslide(model, path, rgb_img_list, dem_img_list, input_tensor_list):
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # 文件夹名称
    folder_name = '/home/lab1015/programmes/guang/code/beng/landslide/img/cam/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_name)
    targets = None
    target_layers_fusion = [model.lambdaLayer]
   # target_layers_fusion = [model.fu1.residual.conv3]

    target_layers_rgb_3 = [model.EB3_0.denseblock[1].conv2d]
    target_layers_dem_3 = [model.EB3_dem.denseblock[1].conv2d]

    target_layers_rgb_2 = [model.EB2_0.denseblock[1].conv2d]
    target_layers_dem_2 = [model.EB2_dem.denseblock[1].conv2d]

    target_layers_rgb_1 = [model.EB1_0.denseblock[1].conv2d]
    target_layers_dem_1 = [model.EB1_dem.denseblock[1].conv2d]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam_fusion = GradCAM(model=model,
                                target_layers=target_layers_fusion,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_3 = GradCAM(model=model,
                                target_layers=target_layers_rgb_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_3 = GradCAM(model=model,
                                target_layers=target_layers_dem_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_2 = GradCAM(model=model,
                                target_layers=target_layers_rgb_2,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_2 = GradCAM(model=model,
                                target_layers=target_layers_dem_2,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    cam_rgb_1 = GradCAM(model=model,
                                target_layers=target_layers_rgb_1,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )
    cam_dem_1 = GradCAM(model=model,
                                target_layers=target_layers_dem_1,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    # cam.batch_size = 32

    for i in range(len(rgb_img_list)):


        
        grayscale_cam_rgb_3 = cam_rgb_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_3 = cam_dem_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )

        grayscale_cam_rgb_2 = cam_rgb_2(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_2 = cam_dem_2(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )

        grayscale_cam_rgb_1 = cam_rgb_1(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_dem_1 = cam_dem_1(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        grayscale_cam_fu = cam_fusion(input_tensor=input_tensor_list[i],
                            targets=targets,
                            #eigen_smooth=args.eigen_smooth,
                            # aug_smooth=args.aug_smooth
                            )
        grayscale_cam_fu = grayscale_cam_fu[0, :]
        
        grayscale_cam_rgb_3 = grayscale_cam_rgb_3[0, :]
        grayscale_cam_dem_3 = grayscale_cam_dem_3[0, :]
        grayscale_cam_rgb_2 = grayscale_cam_rgb_2[0, :]
        grayscale_cam_dem_2 = grayscale_cam_dem_2[0, :]
        grayscale_cam_rgb_1 = grayscale_cam_rgb_1[0, :]
        grayscale_cam_dem_1 = grayscale_cam_dem_1[0, :]

        rgb_img = rgb_img_list[i]
        # dem_img = dem_img_list[i]
        
        path_1 = path
        cv2.imwrite(f'{folder_name}/{i}_rgb.jpg', rgb_img)
        # cv2.imwrite(f'{folder_name}/{i}_dem.jpg', dem_img)
        
        rgb_img = rgb_img / 255.0
        #dem_img = dem_img / 255.0
        
        cam_image_fu = show_cam_on_image(rgb_img, grayscale_cam_fu)
        
        cam_image_rgb_3 = show_cam_on_image(rgb_img, grayscale_cam_rgb_3)
        cam_image_dem_3 = show_cam_on_image(rgb_img, grayscale_cam_dem_3)
        cam_image_rgb_2 = show_cam_on_image(rgb_img, grayscale_cam_rgb_2)
        cam_image_dem_2 = show_cam_on_image(rgb_img, grayscale_cam_dem_2)
        cam_image_rgb_1 = show_cam_on_image(rgb_img, grayscale_cam_rgb_1)
        cam_image_dem_1 = show_cam_on_image(rgb_img, grayscale_cam_dem_1)

        cv2.imwrite(f'{folder_name}/{i}_fusion.jpg', cam_image_fu)
        
        cv2.imwrite(f'{folder_name}/{i}_rgb_layer_3.jpg', cam_image_rgb_3)
        cv2.imwrite(f'{folder_name}/{i}_dem_layer_3.jpg', cam_image_dem_3)
        cv2.imwrite(f'{folder_name}/{i}_rgb_layer_2.jpg', cam_image_rgb_2)
        cv2.imwrite(f'{folder_name}/{i}_dem_layer_2.jpg', cam_image_dem_2)
        cv2.imwrite(f'{folder_name}/{i}_rgb_layer_1.jpg', cam_image_rgb_1)
        cv2.imwrite(f'{folder_name}/{i}_dem_layer_1.jpg', cam_image_dem_1)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_2_on_dem.jpg', cam_image_dem_2_on_dem)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_1_on_dem.jpg', cam_image_dem_1_on_dem)


def print_cam_landslide_res(model, path, rgb_img_list, dem_img_list, input_tensor_list):
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # 文件夹名称
    folder_name = '/home/lab1015/programmes/guang/code/beng/landslide/img/cam/'+path

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_name)
    targets = None
   # target_layers_fusion = [model.fu1.residual.conv3]

    target_layers_rgb_3 = [model.layer4[2].conv3]


    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.

    cam_rgb_3 = GradCAM(model=model,
                                target_layers=target_layers_rgb_3,
                                # use_cuda=True,
                                # reshape_transform=reshape_transform
                                )

    for i in range(len(rgb_img_list)):


        
        grayscale_cam_rgb_3 = cam_rgb_3(input_tensor=input_tensor_list[i],
                            targets=targets,
                            )
        
        grayscale_cam_rgb_3 = grayscale_cam_rgb_3[0, :]

        rgb_img = rgb_img_list[i]
        # dem_img = dem_img_list[i]
        
        path_1 = path
        cv2.imwrite(f'{folder_name}/{i}_rgb.jpg', rgb_img)
        # cv2.imwrite(f'{folder_name}/{i}_dem.jpg', dem_img)
        
        rgb_img = rgb_img / 255.0
        # dem_img = dem_img / 255.0
        
        cam_image_rgb_3 = show_cam_on_image(rgb_img, grayscale_cam_rgb_3)
        
        cv2.imwrite(f'{folder_name}/{i}_rgb_layer_3.jpg', cam_image_rgb_3)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_2_on_dem.jpg', cam_image_dem_2_on_dem)
        # cv2.imwrite(f'./img/test/test_{i}_cam_dem_1_on_dem.jpg', cam_image_dem_1_on_dem)