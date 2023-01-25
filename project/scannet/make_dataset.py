import time
import subprocess
import os
import tqdm
import sys
import glob
import numpy as np
import random



def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list


# 書き換える
cat_name = 'chair'
scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/train.txt')
out_dir = 'results_chair_val' 
scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/val.txt')
# out_dir = 'results_chair_tes' 
# scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/test.txt')
# scene_list = ['scene0477_00', 'scene0477_01'] # txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/results_train/re_check_scene_id_only.txt')
# scene_list = ['scene0093_01','scene0341_01','scene0547_01','scene0592_00','scene0603_00']
scene_list = ['scene0591_00', 'scene0050_00', 'scene0406_00', 'scene0406_01', 'scene0406_02']
out_dir = 'results' 
max_process = 1


#########################
#    start rendering    #
#########################
proc_list = []
sfs_cnt = 0 + 10
devnull = open('/dev/null', 'w')
start_time = time.time()
print('start')
for scene_cnt, scene_name in enumerate(scene_list):
    subprocess_idx = scene_cnt%max_process
    # import pdb; pdb.set_trace()
    proc = subprocess.Popen(
        ['python', 'get_data.py', cat_name, out_dir, scene_name], )
        # stdout=devnull, stderr=devnull)
    proc_list.append(proc)

    if (scene_cnt+1) % max_process == 0 or (scene_cnt+1) == len(scene_list):
        for subproc in proc_list:
            subproc.wait()
        proc_list = []

        # Get time
        elapsed_time = int(time.time() - start_time)
        start_time = time.time()
        elapsed_h = str(elapsed_time // 3600).zfill(2)
        elapsed_min = str((elapsed_time % 3600) // 60).zfill(2)
        elapsed_sec = str((elapsed_time % 3600 % 60)).zfill(2)
        print('COUNT is {}/{} ::: {}:{}:{}'.format(scene_cnt+1, len(scene_list), elapsed_h, elapsed_min, elapsed_sec))
