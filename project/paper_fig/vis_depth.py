import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import glob
import datetime
import cv2
import cv2
import tqdm

sys.path.append("../")
from often_use import *




H = 128
target_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/depth_map/raw_gt_est'
result_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/depth_map/result'
pickle_list = glob.glob(target_dir + '/*')
dt_now = datetime.datetime.now()
time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(os.path.join(result_dir, time_log))

for list_i in tqdm.tqdm(pickle_list):
    str_id = list_i.split('/')[-1].split('.')[0]

    gt, est = pickle_load(list_i)
    mask_min_d = 0
    mask_max_d = 2
    gt_mask = np.logical_and(mask_max_d > gt, gt > mask_min_d)
    est_mask = np.logical_and(mask_max_d > est, est > mask_min_d)
    gt[np.logical_not(gt_mask)] = 0.
    est[np.logical_not(est_mask)] = 0.

    #############################################
    # GtEst
    #############################################

    map_i = np.stack([gt, est]).reshape(-1, H)
    mask_i = np.stack([gt_mask, est_mask]).reshape(-1, H)
    min_d = np.sort(map_i[mask_i])[0]
    max_d = np.sort(map_i[mask_i])[-1]
    map_i[np.logical_not(mask_i)] = min_d
    map_i =(map_i - min_d) / (max_d - min_d)
    map_i = map_i.clip(0, 1)
    # min_d = map_i[mask_i].min() * 0.8
    # max_d = map_i[mask_i].max()
    # map_i = (map_i - min_d) / (max_d - min_d)
    # map_i = map_i.clip(0, 1)

    fig = pylab.figure(figsize=[5,10])
    sns.heatmap(map_i, cbar=False, xticklabels=False, yticklabels=False, cmap='PuBu') # 
    sns.heatmap(map_i, cbar=False, xticklabels=False, yticklabels=False, cmap='viridis') # 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig('tmp.png', dpi=100)
    pylab.close()

    png = cv2.imread('tmp.png')
    os.remove('tmp.png')
    # png = (png - 0 * np.array([[[55, 55, 55]]])).clip(0, 255)
    plt_H = png.shape[0] // 2
    path_gt = os.path.join(result_dir, time_log, 'gt_' + str_id + '.png')
    path_est = os.path.join(result_dir, time_log, 'est_' + str_id + '.png')
    cv2.imwrite(path_gt,  png[:plt_H])
    cv2.imwrite(path_est, png[plt_H:])

    #############################################
    # Dif
    #############################################
    map_i = np.abs(gt - est)
    fig = pylab.figure(figsize=[5,5])
    # sns.heatmap(map_i, center=0, cbar=False, xticklabels=False, yticklabels=False, cmap='RdBu_r')
    sns.heatmap(map_i, cbar=False, xticklabels=False, yticklabels=False, cmap='viridis')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig('tmp.png', dpi=100)
    pylab.close()

    png = cv2.imread('tmp.png')
    # png_m = png.mean()
    # png = (png_m + 0.7 * (png - png_m) - np.array([[[56, 56, 55]]])).clip(0, 255).astype('uint8')
    path_dif = os.path.join(result_dir, time_log, 'dif_' + str_id + '.png')
    cv2.imwrite(path_dif,  png)

    # import pdb; pdb.set_trace()

    # #############################################
    # # Test
    # #############################################
    # aaa = 0.5
    # star_color = np.array([[[150, 255,   0]]])
    # mid_color  = np.array([[[255,   0,   0]]])
    # end_color  = np.array([[[ 70,  10,  35]]])

    # ratio = (map_i - aaa) * (1.0 / (1.0 - aaa))
    # result_map_i = (ratio * star_color + (1 - ratio) * mid_color)

    # mask_aaa = np.tile(map_i < aaa, (1, 1, 3))
    # ratio = map_i * (1.0 / aaa)
    # result_map_i[mask_aaa] = (ratio * mid_color + (1 - ratio) * end_color)[mask_aaa]

    # result_map_i = result_map_i.astype('uint8')
    # cv2.imwrite('tes.png', result_map_i)

