import os
from DeepSDF.project.scannet.old.SensorData import SensorData
import shutil



scene_name = 'scene0705_02'
filename = '/d/workspace/yyoshitake/ScanNet/scans/scene0705_02/scene0705_02.sens'
output_path = 'scannet/data2/scans'

# Dir を作成
scene_out_path = os.path.join(output_path, scene_name)
if not os.path.exists(scene_out_path):
    os.mkdir(scene_out_path)

# 解凍
sd = SensorData(filename)
sd.export_depth_images(os.path.join(scene_out_path, 'depth'))
sd.export_color_images(os.path.join(scene_out_path, 'color'))
sd.export_poses(os.path.join(scene_out_path, 'pose'))
sd.export_intrinsics(os.path.join(scene_out_path, 'intrinsic'))
