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

from parser import *
from often_use import *



test_sample_num = 256
kmean_list_path = '/home/yyoshitake/works/DeepSDF/project/instance_lists/kmean/kmeans_list_2.txt'
kmean_list = []
with open(kmean_list_path, 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        kmean_list.append(line.rstrip('\n'))

test_split = random.sample(kmean_list, test_sample_num)
path = f'kmeans2_test_list.txt'
with open(path, 'a') as f:
    for instance_id in test_split:
        f.write(instance_id+'\n')

train_split = list(set(kmean_list) - set(test_split))
path = f'kmeans2_train_list.txt'
with open(path, 'a') as f:
    for instance_id in train_split:
        f.write(instance_id+'\n')

import pdb; pdb.set_trace()