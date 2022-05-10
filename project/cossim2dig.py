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

from ResNet import *
from parser import *
from dataset import *
from often_use import *
from DDF.train_pl import DDF



target_metrics = {'avg_err_axis_green', 'avg_err_axis_red'}
lines = []
with open('old_experiments_results.txt', 'r') as f:
    file_data = f.readlines()
    for line in file_data:
        line_list = line.split(' ')
        if line_list[0] in target_metrics:
            metrics_value = float(line_list[-1].split('\n')[0])
            metrics_value_dig = math.degrees(math.acos(1-metrics_value))
            lines.append(f'{line_list[0]} : {metrics_value_dig}\n')
        else:
            lines.append(line)

with open('old_experiments_digresults.txt', 'w') as f:
    for line in lines:
        f.write(line)


        