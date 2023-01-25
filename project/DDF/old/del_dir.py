txt_file = '/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/chair/miniddf_train_list1000.txt'
result_list = []
with open(txt_file, 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        result_list.append(line.rstrip('\n'))
result_list = set(result_list)


import time
for i in range(10000):
    # time.sleep(1000)

    import os
    import glob
    base_path = '/disks/local/yyoshitake/ddf/chair/squashfs-root'
    ins_dir_list = glob.glob(os.path.join(base_path, '*'))

    import shutil
    # time.sleep(30)
    for path_i in ins_dir_list:
        ins_name = path_i.split('/')[-1]
        if not ins_name in result_list:
            length = len(glob.glob(os.path.join(path_i, '*')))
            print(length)
            if 400 == length:
                print(path_i)
                shutil.rmtree(path_i)