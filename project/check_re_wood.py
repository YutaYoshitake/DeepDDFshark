import os
import pdb
import sys
from turtle import pd
import numpy as np
import random
import pylab
import glob
import math
import re
from tqdm import tqdm, trange
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.autograd import grad
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

from ResNet import *
from parser import *
from dataset import *
from often_use import *
from train_initnet import *
from train_dfnet import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if device=='cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



ins = instanceSegmentation()
pkl_path = 'pixellib_pkl/pointrend_resnet50.pkl'
ins.load_model(pkl_path)
image_path = 'pixellib_pkl/test_image/0000001-000000000000.jpg'
segmentation_results, image = ins.segmentImage(image_path, show_bboxes=False)
mask_area = [mask_i.sum() for mask_i in segmentation_results['masks'].transpose(2, 0, 1)]

max_idx = mask_area.index(max(mask_area))
print(segmentation_results['class_names'][max_idx])

import pdb; pdb.set_trace()




# depth_path = 'pixellib_pkl/test_image/0000001-000000000000.png'
# depth_map = cv2.imread(depth_path, -1)
# depth_map = (1e4 * cv2.GaussianBlur(depth_map,(3,3),0) / depth_map.max()).astype('uint16')
# edge_mask = np.abs(cv2.Laplacian(depth_map, cv2.CV_32F)) > 1e2

# # segmentation_mask[edge_mask] = False
# # check_map_np(edge_mask, 'dep_map.png')
# check_map_np(segmentation_mask, 'dep_map.png')

# brrured_mask = cv2.GaussianBlur(segmentation_mask.astype('uint8'),(3,3),0)
# _thre, binary = cv2.threshold(brrured_mask, 0.5, 1, cv2.THRESH_BINARY)

# label, contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# for label_i in label:
#     # segmentation_mask[label_i]
#     import pdb; pdb.set_trace()


# depth_path = 'pixellib_pkl/test_image/0000001-000000000000.png'
# org_image = cv2.imread(depth_path, -1)
# org_image = (255*(org_image/org_image.max())).astype('uint8')
# edge_image = cv2.Canny(org_image, 50, 150)
# # kernel = np.ones((3, 3), np.uint8)
# # edge_image = cv2.dilate(edge_image, kernel, iterations=3)
# contours, hierachy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# # check_map_np(edge_image)
# # #描画

# import cv2
# import numpy as np
# from often_use import *

# depth_path = 'pixellib_pkl/test_image/0000001-000000000000.png'
# org_image = cv2.imread(depth_path, -1)
# org_image = (255*(org_image/org_image.max())).astype('uint8')
# edge_image = cv2.Canny(org_image, 50, 150)
# kernel = np.ones((3, 3), np.uint8)
# edge_image = cv2.dilate(edge_image, kernel, iterations=2)

# # 
# segmentation_mask = segmentation_results['masks'][:, :, max_idx]
# # segmentation_mask = segmentation_mask.astype('uint8')
# # segmentation_mask = cv2.GaussianBlur(segmentation_mask,(3,3),0) > 1e-2
# edge_image[np.logical_not(segmentation_mask)] = 255
# check_map_np(edge_image)
# # check_map_np(segmentation_mask )

# contours, hierachy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     import pdb; pdb.set_trace()
#     if len(contour) > 0:

#         # # remove small objects
#         # if cv2.contourArea(contour) < 100:
#         #     continue

#         mask_i = np.zeros_like(org_image)
#         cv2.fillConvexPoly(mask_i, contour, (255))
#         cv2.imwrite('tulips_boundingbox.jpg', mask_i)
