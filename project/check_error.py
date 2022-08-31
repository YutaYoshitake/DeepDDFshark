import os
import sys
from turtle import pd
import numpy as np
import random
import pylab
import glob
import math
import shutil
import re
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from chamferdist import ChamferDistance
from parser import *
from often_use import *





label_a = "trans_avg"
label_b = "baseline"
aaa_date = '2022_08_30_06_35_54'
bbb_date = '2022_08_30_06_35_10'
aaa = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{aaa_date}/log_error.pickle')
bbb = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{bbb_date}/log_error.pickle')
data_mode = 'randn'
target_mode = f'trans_is_better'
parent_directory_path = f'sample_images/list0randn/{data_mode}/{target_mode}/'
os.makedirs(parent_directory_path, exist_ok=True)

# label_a = "trans_avg"
# label_b = "baseline"
# aaa_date = '2022_08_30_03_24_04'
# bbb_date = '2022_08_30_01_39_42'
# aaa = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{aaa_date}/log_error.pickle')
# bbb = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{bbb_date}/log_error.pickle')
# data_mode = 'fixed'
# target_mode = f'trans_is_better'
# parent_directory_path = f'sample_images/list0randn/{data_mode}/{target_mode}/'
# os.makedirs(parent_directory_path, exist_ok=True)

# for key in aaa.keys():
#     # if key in {'pos', 'green', 'red', 'scale', 'depth'}:
#     if key in {'depth'}:
#         fig = pylab.figure()
#         ax = fig.add_subplot(1,1,1)
#         ax.hist([
#             aaa[key].squeeze(), 
#             bbb[key].squeeze()], bins=50, label=[label_a, label_b], log=True)
#         ax.legend()
#         fig.savefig(f"err_{key}_{data_mode}.png")
# import pdb; pdb.set_trace()


# # bad_targetのエラーが大きく、
# # better_targetのエラーが小さい例。
# key = 'red'
# bad_target = bbb # base
# better_target = aaa # tra
# bad_model_min = 10
# better_model_max = 3

# # bad_targetだけ悪い。
# min_mask = bad_target[key] > bad_model_min 
# max_mask = better_target[key][min_mask] < better_model_max

# # 両方悪い。
# # min_mask = bad_target[key] > bad_model_min 
# # max_mask = better_target[key][min_mask] > bad_model_min

# path_list = better_target['path'][min_mask][max_mask]
# rand_P_seed = better_target['rand_P_seed'][min_mask][max_mask]
# rand_S_seed = better_target['rand_S_seed'][min_mask][max_mask]
# randn_theta_seed = better_target['randn_theta_seed'][min_mask][max_mask]
# randn_axis_idx = better_target['randn_axis_idx'][min_mask][max_mask]

# print(len(path_list))
# bad_target[key][min_mask][max_mask]
# better_target[key][min_mask][max_mask]
# sum(bad_target[key] > bad_model_min)
# sum(better_target[key] > bad_model_min)

# for path_i in path_list:
#     split_dot = path_i.split('.')[0] # '738395f54b301d80b1f5d603f931c1aa/00001'
#     split_slash = split_dot.split('/') # '738395f54b301d80b1f5d603f931c1aa', '00001'
#     path_i = f'dataset/dugon/moving_camera/train/kmean0_{data_mode}/' + split_dot + '.png'
#     target_name = path_i.split('.')[0]
#     shutil.copyfile(path_i, parent_directory_path + split_slash[0] + '_' + split_slash[1] + '.png')
# import pdb; pdb.set_trace()


# pickle_name = 'Rad.pickle'
# data_dict = {'path':path_list, 'rand_P_seed':rand_P_seed, 'rand_S_seed':rand_S_seed, 'randn_theta_seed':randn_theta_seed, 'randn_axis_idx':randn_axis_idx}
# pickle_dump(data_dict, f'000_{target_mode}_{target_mode}.pickle')
# import pdb; pdb.set_trace()


aaa_avg_log = {
    'pos' : np.array([0.15056264, 0.041626837, 0.020613406, 0.01565276, 0.01430345, 0.013998766]), 
    'green' : np.array([35.05945, 9.837852, 4.0477667, 2.9562793, 2.784236, 2.777223]), 
    'red' : np.array([34.601006, 14.917136, 5.514079, 3.2407093, 2.7908962, 2.6284115]), 
    'scale' : np.array([0.14908247, 0.09709674, 0.045342874, 0.043955114, 0.04671287, 0.048863348]), 
    'shape' : np.array([0.11309633, 0.11423056, 0.09505049, 0.08657477, 0.08538977, 0.08539624]), 
    }


aaa_med_log = {
    'pos' : np.array([0.15141244, 0.037736975, 0.018097376, 0.013762844, 0.012551535, 0.012287206]), 
    'green' : np.array([32.443924, 7.987003, 3.3821697, 2.5984201, 2.5304513, 2.5487561]), 
    'red' : np.array([31.590153, 10.900149, 3.5682874, 2.0715504, 1.7912761, 1.6784723]), 
    'scale' : np.array([0.14883542, 0.075674, 0.038639307, 0.038916737, 0.042843014, 0.04523912]), 
    'shape' : np.array([0.114758685, 0.11213647, 0.09303734, 0.08508292, 0.08364142, 0.083610624]), 
    }

bbb_avg_log = {
    'pos' : np.array([0.15056264, 0.05596557, 0.031111112, 0.0209413, 0.016729988, 0.014851256]), 
    'green' : np.array([35.05945, 15.122427, 6.305102, 3.245079, 2.244779, 1.9063562]), 
    'red' : np.array([34.601006, 20.18334, 9.8631115, 5.8022523, 4.1355586, 3.3885996]), 
    'scale' : np.array([0.14908247, 0.16531149, 0.0919075, 0.060989607, 0.050514653, 0.04767442]), 
    'shape' : np.array([0.11309633, 0.11850956, 0.101638004, 0.0911587, 0.0863531, 0.08443022]), 
    }

bbb_med_log = {
    'pos' : np.array([0.15141244, 0.04883195, 0.025358409, 0.016685855, 0.013490843, 0.012084553]), 
    'green' : np.array([32.443924, 11.463316, 4.1068416, 2.023859, 1.5544411, 1.4232655]), 
    'red' : np.array([31.590153, 14.074621, 5.0661416, 2.4849858, 1.8294358, 1.6376343]), 
    'scale' : np.array([0.14883542, 0.13335133, 0.057513297, 0.039941877, 0.038881034, 0.038632244]), 
    'shape' : np.array([0.114758685, 0.115968704, 0.09927876, 0.08843007, 0.084171735, 0.082229]), 
    }

itr = np.array([1, 2, 3, 4, 5])
for key in aaa_avg_log.keys():
    if key in {'pos', 'green', 'red', 'scale', 'shape'}:
        fig = pylab.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(itr, aaa_med_log[key].squeeze()[1:], label=label_a)
        ax.plot(itr, bbb_med_log[key].squeeze()[1:], label=label_b)
        # plt.yscale("log")
        ax.set_xlabel('iteration')
        ax.set_ylabel('error')
        ax.legend()
        fig.savefig(f"log_{key}_{data_mode}.png")







# origin = aaa
# target = bbb
# result_path_list = []
# original_value_list = []
# target_value_list = []
# err_key = 'red'

# target_idx_list = np.flipud(np.argsort(origin[err_key])) # 誤差の大きい順で並べる
# origin_threshold = 70
# original_num = 128
# mask = origin[err_key][target_idx_list] > origin_threshold
# target_idx_list = target_idx_list[mask][:original_num] # しきい値以上のインデックスを取得
# # print(origin[err_key][target_idx_list]) # エラーを表示

# threshold = 5
# for idx in target_idx_list:
#     origin_value = origin[err_key][idx]
#     target_value = target[err_key][idx]
#     if target_value < threshold:
#         if target_value < origin_value:
#             result_path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/' + origin['path'][idx]
#             result_path_list.append(result_path)
#             print(f'origin : {origin[err_key][idx]:.1f}, target : {target[err_key][idx]:.1f}')
#             # original_value_list.append(origin[err_key][idx])
#             # target_value_list.append(target[err_key][idx])

# # print(original_value_list)
# # print(target_value_list)
# print('###     ' + str(len(result_path_list)) + '     ###')

# pickle_dump(result_path_list, pickle_name)
# import pdb; pdb.set_trace()



# mmm = []
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_05/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_11/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_14/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_18/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_21/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_24/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_27/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_31/log.pickle'))

# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_27/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_32/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_35/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_40/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_43/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_46/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_49/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_52/log_error.pickle'))

# ooo = {}
# for key in mmm[0].keys():
#     nnn = []
#     for mmm_i in mmm:
#         nnn.append(mmm_i[key])
#     nnn = np.concatenate(nnn)
#     ooo[key] = nnn
# pickle_dump(ooo, f'adam_ind_update.pickle')
# import pdb; pdb.set_trace()



# df = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_06_08_23_07_18/log_error.pickle')
# adam = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_06_07_12_16_00/log_error.pickle')

# for key in df.keys():
#     # x = df[key]
#     # y = adam[key]

#     # fig = pylab.figure()
#     # ax = fig.add_subplot(1,1,1)
#     # ax.set_title(f'{key}')
#     # ax.hist(x, bins=50)
#     # ax.hist(y, bins=50)
#     # fig.savefig(f"err_{key}.png")

#     if key != 'path':
#         for path in enumerate(adam['path'][np.argsort(adam[key])]):
#             with open(f'adam_{key}.txt', 'a') as f:
#                 f.writelines(f'{path[-1]}\n')
        
#         for path in enumerate(df['path'][np.argsort(df[key])]):
#             with open(f'df_{key}.txt', 'a') as f:
#                 f.writelines(f'{path[-1]}\n')
#     import pdb; pdb.set_trace()



# pickle_name = 'adam_vs_deep_kmeans0_test.pickle'
# mmm = pickle_load('/home/yyoshitake/works/DeepSDF/project/adam_vs_deep_kmeans0_test.pickle')
# for idx in range(len(mmm)):
#     mmm[idx] = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/' + mmm[idx]
# pickle_dump(mmm, pickle_name)