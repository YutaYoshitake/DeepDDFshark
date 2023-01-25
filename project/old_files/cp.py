import os
import shutil

def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list

ins_list = txt2list('/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/val.txt')
for ins in ins_list:
    shutil.copyfile(f"/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical/{ins}.pickle", 
    f"/disks/local/yyoshitake/moving_camera/canonical/tmp_map/03001627/{ins}.pickle")