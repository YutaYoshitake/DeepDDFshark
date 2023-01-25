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
from PIL import Image
from parser import *
from often_use import *





label_a = "OnlyMLP"
label_b = "Ours wo Dec" # "Ours (tau = 2)"
# label_b = "autoreg"
aaa_date = '2022_11_21_06_55_03_subrndnsim_onlymlp_tau00_epo0000000792' # '2022_11_20_08_35_18_subrndnsim_encoder_tau00_epo0000000800'
bbb_date = '2022_11_20_08_35_18_subrndnsim_encoder_tau00_epo0000000800' # '2022_11_20_08_34_42_subrndnsim_autoreg_tau01_epo0000000800'
ccc_date = '2022_11_20_15_28_58_subrndnsim_autoreg_tau02_epo0000000752'
aaa = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{aaa_date}/log_error.pickle')
bbb = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{bbb_date}/log_error.pickle')
bbb = pickle_load(f'/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/{ccc_date}/log_error.pickle')
data_mode = 'tes'
target_mode = f'dec_is_bad'
parent_directory_path = f'sample_images/list0randn/{data_mode}/{target_mode}/'
os.makedirs(parent_directory_path, exist_ok=True)



x_label = {'pos': 'Translation error', 
           'rot': 'Riemannian distance', 
           'chm_ss_obj': 'Chamfer distance', 
           'green': 'Green axis error (Deg)', 
           'red': 'Red axis error (Deg)', 
           'scale': 'Scale error (%)', 
           'shape': 'Depth error', }
for key in ['pos', 'dig', 'chm_ss_obj']:
    if key == 'dig':
        for err_list in [aaa, bbb]:
            err_list['dig'] = np.stack([err_list['red'], err_list['green']], axis=0).max(axis=0)
            x_label['dig'] = 'Max axis error (Deg)'
    fig = pylab.figure(figsize=(4.5,3.5))
    ax = fig.add_subplot(1,1,1)
    ax.hist([
        aaa[key][:, -1].squeeze(), 
        bbb[key][:, -1].squeeze(),], bins=50, label=[label_a, label_b], log=True)
    ax.legend()
    ax.set_xlabel(x_label[key])
    ax.set_ylabel('Frequency')
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(left=0.137)
    fig.subplots_adjust(bottom=0.137)
    fig.savefig(f"err_{data_mode}_{key}.png")
plt_im_list = []
for key in ['pos', 'dig', 'chm_ss_obj']:
    plt_im_list.append(np.array(Image.open(f"err_{data_mode}_{key}.png")))
plt_im = Image.fromarray(np.concatenate(plt_im_list, axis=1))
plt_im.save('tes.png')
import pdb; pdb.set_trace()



# label_a = "wo/Decoder (tau = 0)"
# label_b = "AutoReg (tau = 1)"
# label_c = "AutoReg (tau = 2)"
# y_label = {'est_p': 'Translation', 
#            'est_g': 'Green axis', 
#            'est_r': 'Red axis', 
#            'est_s': 'Scale', 
#            'est_z': 'Shape', }
# for key_i in ['est_p', 'est_g', 'est_r', 'est_s', 'est_z']:
#     fig = pylab.figure(figsize=(4.5,3.5))
#     ax = fig.add_subplot(1,1,1)
#     for log_i, label_i in zip([aaa, bbb, ccc], [label_a, label_b, label_c]):
#         dif_norm = np.linalg.norm((log_i[key_i][:-1] - log_i[key_i][1:]), axis=-1).mean(-1)
#         ax.plot(np.arange(dif_norm.shape[0]), 100*dif_norm/dif_norm[0], label=label_i)
#     ax.set_title(y_label[key_i])
#     ax.set_xlabel('Steps')# ('Time (Sec)')
#     ax.set_ylabel('Update norm (%)')
#     ax.legend()
#     fig.subplots_adjust(right=0.95)
#     fig.subplots_adjust(top=0.9)
#     fig.subplots_adjust(left=0.16)
#     fig.subplots_adjust(bottom=0.138)
#     fig.savefig(f"{key_i}.png", dpi=300)
# plt_im_list = []
# for key_i in ['est_p', 'est_g', 'est_r', 'est_s', 'est_z']:
#     plt_im_list.append(np.array(Image.open(f"{key_i}.png")))
# plt_im = Image.fromarray(np.concatenate(plt_im_list, axis=1))
# plt_im.save('tes.png')
# import pdb; pdb.set_trace()





# # bad_targetのエラーが大きく、
# # better_targetのエラーが小さい例。
# key = 'red'
# bad_target = aaa # base
# better_target = bbb # tra
# bad_model_min = 0.5
# better_model_max = 180

# # bad_targetだけ悪い。
# min_mask = bad_target[key] < bad_model_min 
# max_mask = better_target[key][min_mask] < better_model_max

# # 両方悪い。
# # min_mask = bad_target[key] > bad_model_min 
# # max_mask = better_target[key][min_mask] > bad_model_min

# path_list = better_target['path'][min_mask][max_mask]
# rand_P_seed = better_target['rand_P_seed'][min_mask][max_mask]
# rand_S_seed = better_target['rand_S_seed'][min_mask][max_mask]
# randn_theta_seed = better_target['randn_theta_seed'][min_mask][max_mask]
# randn_axis_idx = better_target['randn_axis_idx'][min_mask][max_mask]

# target_idx = np.argsort(bbb['shape'])[50:100][12]
# path_list = [bbb['path'][target_idx]]
# rand_P_seed = [bbb['rand_P_seed'][target_idx]]
# rand_S_seed = [bbb['rand_S_seed'][target_idx]]
# randn_theta_seed = [bbb['randn_theta_seed'][target_idx]]
# randn_axis_idx = [bbb['randn_axis_idx'][target_idx]]

# print(len(path_list))
# bad_target[key][min_mask][max_mask]
# better_target[key][min_mask][max_mask]
# sum(bad_target[key] > bad_model_min)
# sum(better_target[key] > bad_model_min)

# par = []
# for path_i in path_list:
#     split_dot = path_i.split('.')[0] # '738395f54b301d80b1f5d603f931c1aa/00001'
#     split_slash = split_dot.split('/') # '738395f54b301d80b1f5d603f931c1aa', '00001'
#     # path_i = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/kmean_list0/kmean0_randn/resolution128/raw/' + split_dot + '.png'
#     # target_name = path_i.split('.')[0]
#     # shutil.copyfile(path_i, parent_directory_path + split_slash[0] + '_' + split_slash[1] + '.png')
#     # path_i = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/kmean_list0/kmean0_randn/resolution128/raw' + split_dot + '.pickle'
#     # par.append(pickle_load(path_i)['mask'].sum() / (3*128*128))
# # print(sum(par)/len(par))
# import pdb; pdb.set_trace()


# pickle_name = 'Rad.pickle'
# data_dict = {'path':path_list, 'rand_P_seed':rand_P_seed, 'rand_S_seed':rand_S_seed, 'randn_theta_seed':randn_theta_seed, 'randn_axis_idx':randn_axis_idx}
# pickle_dump(data_dict, f'000_{target_mode}_{target_mode}.pickle')
# import pdb; pdb.set_trace()
