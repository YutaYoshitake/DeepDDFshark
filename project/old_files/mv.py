import os
import shutil

def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list

train_ins_list = txt2list('/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/fultrain.txt')
val_ins_list   = txt2list('/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/val.txt')
import pdb; pdb.set_trace()
ins_list = train_ins_list + val_ins_list
for ins in ins_list:
    shutil.move(f"/home/yyoshitake/works/DeepSDF/disks/old/chair/tmp_2/results/{ins}", 
    f"/home/yyoshitake/works/DeepSDF/disks/old/chair/tmp_2/result_disks")