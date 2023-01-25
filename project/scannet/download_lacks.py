import time
import subprocess
import glob



def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list



downloaded_list = glob.glob('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scans/raw/*')
downloaded_list = [path_i.split('/')[-1] for path_i in downloaded_list]

tes_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/test.txt')
train_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/train.txt')
val_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/val.txt')
total_list = tes_list + val_list + train_list
print(len(total_list))
# import pdb; pdb.set_trace()
total_list = ['scene0591_00', 
              'scene0050_00', 
              'scene0406_00', 
              'scene0406_01', 
              'scene0406_02', ]

#########################
#    start rendering    #
#########################
max_process = 1
proc_list = []
sfs_cnt = 0 + 10
devnull = open('/dev/null', 'w')
start_time = time.time()
print('start')
for scene_cnt, scene_name in enumerate(total_list):
    print(scene_name)
    if True: #  not scene_name in downloaded_list:
        subprocess_idx = scene_cnt%max_process
        # import pdb; pdb.set_trace()
        # python download-scannet.py -o scan_tes --id scene0000_00
        proc = subprocess.Popen(
            ['python', 'download-scannet.py', '-o', 'scan_tes', '--id', scene_name], )
            # stdout=devnull, stderr=devnull)
        proc_list.append(proc)

        if (scene_cnt+1) % max_process == 0 or (scene_cnt+1) == len(total_list):
            for subproc in proc_list:
                subproc.wait()
            proc_list = []

            # # Get time
            # elapsed_time = int(time.time() - start_time)
            # start_time = time.time()
            # elapsed_h = str(elapsed_time // 3600).zfill(2)
            # elapsed_min = str((elapsed_time % 3600) // 60).zfill(2)
            # elapsed_sec = str((elapsed_time % 3600 % 60)).zfill(2)
            # print('COUNT is {}/{} ::: {}:{}:{}'.format(scene_cnt+1, len(total_list), elapsed_h, elapsed_min, elapsed_sec))
