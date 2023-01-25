# blender --background --python make_offset_pickle.py

import bpy
import sys

sys.path.insert(0, "/opt/pyenv/versions/3.5.10/lib/python3.5/site-packages/")

import numpy as np
import sys
import pickle
import json
import os



def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)



out_dir = 'results'
ShapeNetPath = '/d/workspace/yyoshitake/ShapeNet/ShapeNetCore.v2'
target_shapenet_cat = {'04379243': 'table', 
                       '03001627': 'chair', 
                       '03211117': 'display', 
                       '02933112': 'cabinet', }

# Scan2CADのアノテーション読み込み
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)
# 辞書型で並進ログを作成
loc_dict = {}
for annotation_i in annotations_json_load:
    scan_id = annotation_i['id_scan']
    if scan_id == 'scene0000_00':
        for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
            catid_cad = cad_i['catid_cad']
            id_cad = cad_i['id_cad']
            if catid_cad in target_shapenet_cat.keys():
                total_id = '{}_{}'.format(catid_cad, id_cad)
                loc_dict[total_id] = 0



scene = bpy.context.scene
camera_distance_range = (1.0, 1.0)

# delete default cube and lamp
bpy.data.objects['Cube'].select = True
bpy.data.objects['Lamp'].select = True
bpy.ops.object.delete() 
camera = bpy.data.objects['Camera']

# config to save depth map
## Set up rendering of depth map:
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

## clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

## create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')

map = tree.nodes.new(type="CompositorNodeMapValue")
## Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
map.use_min = False
map.use_max = False
links.new(rl.outputs[2], map.inputs[0])

invert = tree.nodes.new(type="CompositorNodeInvert")
links.new(map.outputs[0], invert.inputs[1])

## The viewer can come in handy for inspecting the results in the GUI
depthViewer = tree.nodes.new(type="CompositorNodeViewer")
links.new(invert.outputs[0], depthViewer.inputs[0])
## Use alpha from input.
links.new(rl.outputs[1], depthViewer.inputs[1])



# Rendering images.
for cat_ins_id in loc_dict.keys():

    class_name, instance_name = cat_ins_id.split('_')
    
    # Loading model.
    model_dir = ShapeNetPath + '/' + class_name + '/' + instance_name
    bpy.ops.import_scene.obj(filepath=model_dir+'/models/model_normalized.obj', split_mode = "OFF")
    imported = bpy.context.selected_objects[0]
    model = bpy.data.objects[imported.name]
    model.location = (0.0, 0.0, 0.0)
    model.rotation_mode = 'YXZ'

    # Align object center with world coordinates.
    center = np.zeros((3))
    for pt in model.bound_box:
        center += np.array([pt[0], pt[1], pt[2]]) / 8
    loc = np.array(model.location) - center
    
    bpy.ops.object.select_all(action='DESELECT')
    model.select = True
    bpy.ops.object.delete()

    loc_dict[cat_ins_id] = loc

# 保存
pickle_dump(loc_dict, os.path.join(out_dir, 'loc_dict.pickle'))
