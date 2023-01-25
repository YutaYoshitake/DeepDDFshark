import glob
import shutil
import os
base_dir = 'dataset/table/results_cmp'
data_path_list = glob.glob(base_dir + '/*')
data_num = len(data_path_list)
split_num = 5
splitted_data_num = data_num // split_num + 1
for i in range(split_num):
    if not os.path.exists(base_dir + f'_{str(i)}'): os.mkdir(base_dir + f'_{str(i)}')
    for path_idx_split in range(splitted_data_num):
        path_idx = path_idx_split + i * splitted_data_num
        data_path = data_path_list[path_idx]
        print(path_idx)
        shutil.move(data_path, base_dir + f'_{str(i)}')
