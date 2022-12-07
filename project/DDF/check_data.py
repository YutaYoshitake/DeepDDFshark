from often_use import *
import tqdm


lack_normal = []
ins_list = txt2list('/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/instance_list.txt')
for ins in tqdm.tqdm(ins_list):
    data_dict = pickle_load(f'/d/workspace/yyoshitake/ShapeNet/ddf/chair/train_data/{ins}/00000_mask.pickle')
    if not 'normal_map' in data_dict.keys():
        lack_normal.append(ins)

list2txt(lack_normal, 'lack_normal.txt')