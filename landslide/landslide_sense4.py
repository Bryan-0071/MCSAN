import os
import glob
import numpy as np
# from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import rasterio
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import gradient, pi, sin, cos, arctan2, sqrt
from scipy.ndimage import generic_filter
import matplotlib
# from skimage.feature import greycomatrix, greycoprops
import sys
sys.path.append("..")
sys.path.append(".")
from tools.fusion_transformer import Fusion_network_Transformer
from configs.landslide_sense4 import args
import random
from tools.load_data import test_model_in_epoch_sense4
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_process.sense4 import LandslideDataSet

from torch.cuda.amp import GradScaler, autocast
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from tools.cam_function import print_cam_landslide
from torch.utils import data
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from timm.models.resnet import ResNet,resnet50
# from timm.models.swin_transformer import SwinTransformer
# from timm.models.vision_transformer import vit_base_patch16_224
from timm.models.resnest import resnest50d
# from timm.models.hrnet import hrnet_w30,hrnet_w48
# from timm.models.swin_transformer_v2 import swinv2_base_window8_256
# from timm.models.eva import eva02_base_patch14_224
# from timm.models.visformer import visformer_small
# from timm.models.cait import cait_s24_224
# from timm.models.deit import deit_base_patch16_224
# from timm.models.swin_transformer_v2 import swinv2_base_window8_256
# from timm.models.convnext import convnext_base
# from timm.models.focalnet import focalnet_base_lrf, focalnet_base_srf, focalnet_huge_fl3, focalnet_huge_fl4, focalnet_large_fl3, focalnet_large_fl4, focalnet_small_lrf, focalnet_small_srf, focalnet_tiny_lrf, focalnet_tiny_srf, focalnet_xlarge_fl3, focalnet_xlarge_fl4
import sys
import os
from ipdb import set_trace
# from pytorch_image_models.timm.models.resnest import resnest50d
from torch.cuda.amp import GradScaler, autocast
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
# Set random seeds for reproducibility.
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")

# Keep CUDA behavior deterministic when GPUs are available.
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

# Define the size to which all features will be resized
feature_size = args.feature_size
batch_size = args.batch_size

if __name__ == '__main__':
    print("Loading data...")
    train_loader = data.DataLoader(
                    LandslideDataSet(args.data_dir, args.train_list, augment=False),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valid_loader = data.DataLoader(
                        LandslideDataSet(args.data_dir, args.test_list),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = data.DataLoader(
                    LandslideDataSet(args.data_dir, args.test_list),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    

    nb_filter = args.nb_filter
    # Initialize AMP utilities.
    scaler = GradScaler()
    # Number of gradient accumulation steps.
    accumulate_grads = 4

    from net.dense_res_crossfuse import NestFuse_light2_nodense2 as Dense_res_crossfuse
    model = Dense_res_crossfuse(nb_filter, input_nc=3, output_nc=2, deepsupervision=False)

    # model = resnest50d(in_chans=5, num_classes=2)
    torch.cuda.set_device(0)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    print("Model loaded. Training... lr =", args.lr, "wd =", args.wd)
    optimizer = optim.AdamW([*model.parameters()], lr=args.lr, weight_decay=args.wd)
    model.cuda()

    import time
    # Training loop
    time1=time.time()
    num_epochs = args.epochs
    # num_epochs = 1
    best_acc = 0
    best_f1 = 0.0
    path_best_model = None
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for batch_id, src_data in enumerate(train_loader):
            # print("batch_id = ", batch_id)
            model.train()
            images, labels, _, _ = src_data
            
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.squeeze(1)
            labels = labels.long()
            outputs = model(images)
            loss = criterion(outputs,labels).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        train_accuracy = 100 * correct / total
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Validation: compute metrics on valid_loader and save best model if Valid F1 >= 0.7
        model.eval()
        val_y_true = []
        val_y_pred = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, src_data in enumerate(valid_loader):
                images, labels, _, _ = src_data
                images = images.cuda()
                labels = labels.cuda()
                labels = labels.squeeze(1).long()  
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_y_true.extend(labels.cpu().numpy())
                loss_val = criterion(outputs, labels).cuda()
                val_y_pred.extend(predicted.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_accuracy = 100 * correct / total
        val_recall = recall_score(val_y_true, val_y_pred, average='macro')
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {loss_val.item():.4f}, ", end="")
        print(f"Validation Accuracy: {val_accuracy:.2f}%, ", end="")
        print(f"Valid Recall: {val_recall:.4f}, Valid F1: {val_f1:.4f}")

        # Save model if validation F1 improves and is >= 0.7
        if val_f1 > best_f1 and val_f1 >= 0.75:
            best_f1 = val_f1
            save_dir = '/home/ec2-user/code/landslide/landslide/save'
            os.makedirs(save_dir, exist_ok=True)
            model_name = model.__class__.__name__ if hasattr(model, '__class__') else 'model'
            lr_str = str(args.lr).replace('.', '_')
            fname = f'best_my{model_name}_lr{lr_str}_f1{val_f1:.4f}.pth'
            path_best_model = os.path.join(save_dir, fname)
            torch.save(model.state_dict(), path_best_model)
            print(f"Saved best model to {path_best_model}")

        print("Training time:", time.time() - time1)

    path_save = 'best'
    # Test the model
    # If a best model was saved during training, load it for testing
    if path_best_model is not None and os.path.exists(path_best_model):
        print(f"Loading best model from {path_best_model} for testing")
        model.load_state_dict(torch.load(path_best_model))
    model.eval()

    with torch.no_grad():
        time2=time.time()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for batch_id, src_data in enumerate(test_loader):
            # images = (images - mean) / std
            images, labels, _, _ = src_data
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.squeeze(1)
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Test Accuracy: {100 * correct / total:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print("Inference time", time.time() - time2)
        # Confusion matrix
        labels_list = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_list)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        # Save confusion matrix to fixed folder with lr and model name in filename
        save_dir = '/home/ec2-user/code/landslide/landslide/img/matric'
        os.makedirs(save_dir, exist_ok=True)
        model_name = model.__class__.__name__ if hasattr(model, '__class__') else 'model'
        lr_str = str(args.lr).replace('.', '_')
        filename = f'confusion_matrix_{model_name}_lr{lr_str}.png'
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path)
        plt.close()
