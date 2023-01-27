import numpy as np
np.warnings.filterwarnings('ignore')
import pathlib
import subprocess
import os
import collections
import shutil
import quaternion
import operator
import glob
import csv
import re
# import CSVHelper
import json
import pickle
from objectron.box import Box
from objectron.iou import IoU
import argparse
np.seterr(all='raise')
import argparse


def JSONHelper_read(filename):
	with open(filename, 'r') as infile:
		return json.load(infile)


def SE3_compose_mat4(t, q, s, center=None):
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M 


def SE3_decompose_mat4(M):
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    q = quaternion.from_rotation_matrix(R[0:3, 0:3])
    #q = quaternion.from_float_array(quaternion_from_matrix(M, False))

    t = M[0:3, 3]
    return t, q, s


# params
parser = argparse.ArgumentParser()                                                                                                                                                                                                                                                                                        
# parser.add_argument('--dataset', required=True, choices=["scannet" ],help="choose dataset")
parser.add_argument('--projectdir', required=True, help="project directory")
opt = parser.parse_args()
opt.dataset = 'scannet'
opt.tgt_cat_name = 'chair'


# get top8 (most frequent) classes from annotations. 
def get_top8_classes_scannet():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "trashbin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    return top


# helper function to calculate difference between two quaternions 
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def evaluate(projectdir, filename_cad_appearance, filename_annotations):
    appearances_cad = JSONHelper_read(filename_cad_appearance)

    benchmark_per_scan = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_scan
    benchmark_per_class = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_class
    if opt.dataset == "scannet":
        catid2catname = get_top8_classes_scannet()
    
    groundtruth = {}
    cad2info = {}
    idscan2trs = {}
    
    # testscenes = [os.path.basename(f).split(".")[0] for f in glob.glob(projectdir + "/*.csv")]
    testscenes_list_txt = './data2/scan2cad/split/test.txt'
    testscenes_list_gt = []
    with open(testscenes_list_txt, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            testscenes_list_gt.append(line.rstrip('\n'))
    
    testscenes_gt = []
    for r in JSONHelper_read(filename_annotations):
        id_scan = r["id_scan"]
        # NOTE: remove this
        if id_scan not in testscenes_list_gt:
            continue
        # <-
        testscenes_gt.append(id_scan)

        idscan2trs[id_scan] = r["trs"]
        
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
            catname_cad = catid2catname[catid_cad]
            model["n_total"] = len(r["aligned_models"])
            groundtruth.setdefault((id_scan, catid_cad),[]).append(model)
            cad2info[(catid_cad, id_cad)] = {"sym" : model["sym"], "catname" : catname_cad}

            benchmark_per_class[catname_cad]["n_total"] += 1
            benchmark_per_scan[id_scan]["n_total"] += 1

    projectname = os.path.basename(os.path.normpath(projectdir))

    # Iterate through your alignments
    # counter = 0
    # for file0 in glob.glob(projectdir + "/*.csv"):
    #     alignments = CSVHelper.read(file0)
    #     id_scan = os.path.basename(file0.rsplit(".", 1)[0])
    est_scene_paths = glob.glob(os.path.join(opt.projectdir, 'estimations', '*'))
    counter = 0
    for est_scene_path in est_scene_paths:
        id_scan = est_scene_path.split('/')[-1]
    
        if id_scan not in testscenes_gt:
            import pdb; pdb.set_trace()
            continue
        benchmark_per_scan[id_scan]["seen"] = 1

        alignments = glob.glob(os.path.join(est_scene_path, '*'))

        appearance_counter = {}

        for alignment in alignments: # <- multiple alignments of same object in scene
            # -> read from .csv file
            # catid_cad = alignment[0]
            # id_cad = alignment[1]
            catid_cad, id_cad, _ = alignment.split('/')[-1].split('.')[0].split('_')
            cadkey = catid_cad + "_" + id_cad
            #import pdb; pdb.set_trace()
            if cadkey in appearances_cad[id_scan]:
                n_appearances_allowed = appearances_cad[id_scan][cadkey] # maximum number of appearances allowed
            else:
                n_appearances_allowed = 0
            
            appearance_counter.setdefault(cadkey, 0)
            if appearance_counter[cadkey] >= n_appearances_allowed:
                continue
            appearance_counter[cadkey] += 1

            # 自分の推定値を読み込み
            with open(alignment, mode='rb') as f:
                estimation_params = pickle.load(f)
            t_obj2wrd = estimation_params['t']
            q_obj2wrd = np.quaternion(estimation_params['q'])
            s_obj2wrd = estimation_params['s']

            catname_cad = cad2info[(catid_cad, id_cad)]["catname"]
            sym = cad2info[(catid_cad, id_cad)]["sym"]
            # t = np.asarray(alignment[2:5], dtype=np.float64)
            # q0 = np.asarray(alignment[5:9], dtype=np.float64)
            # q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
            # s = np.asarray(alignment[9:12], dtype=np.float64)
            # <-

            key = (id_scan, catid_cad) # <-- key to query the correct groundtruth models
            for idx, model_gt in enumerate(groundtruth[key]):

                is_same_class = model_gt["catid_cad"] == catid_cad # <-- is always true (because the way the 'groundtruth' was created
                if is_same_class: # <-- proceed only if candidate-model and gt-model are in same class

                    # <-- transformation from scan space to world space
                    Mscan = SE3_compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
                    # <-- transformation from CAD space to world space
                    # Mcad = SE3_compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"], -np.array(model_gt["center"])) # bagu ?
                    Mcad = SE3_compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"], np.array(model_gt["center"])) # 俺案

                    # 推定値の結果
                    # from often_use import pickle_load
                    # loc_dict = pickle_load('/home/yyoshitake/works/DeepSDF/project/scannet/loc_dict.pickle')
                    # correct_loc = - loc_dict[f'{catid_cad}_{id_cad}'] == np.array(model_gt["center"])
                    # offset = quaternion.as_rotation_matrix(q_obj2wrd) @ (loc * s_obj2wrd)
                    # SE3_compose_mat4(t_obj2wrd+2*offset, q_obj2wrd, s_obj2wrd) 
                    estMcad = SE3_compose_mat4(t_obj2wrd, q_obj2wrd, s_obj2wrd)
                    
                    # <-- transformation from CAD space to scan space
                    t_gt, q_gt, s_gt = SE3_decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))
                    t, q, s = SE3_decompose_mat4(np.dot(np.linalg.inv(Mscan), estMcad))

                    error_translation = np.linalg.norm(t - t_gt, ord=2)
                    error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)

                    # --> resolve symmetry
                    if sym == "__SYM_ROTATE_UP_2":
                        m = 2
                        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    elif sym == "__SYM_ROTATE_UP_4":
                        m = 4
                        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    elif sym == "__SYM_ROTATE_UP_INF":
                        m = 36
                        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    else:
                        error_rotation = calc_rotation_diff(q, q_gt)

                    # -> define Thresholds
                    threshold_translation = 0.2 # <-- in meter
                    threshold_rotation = 20 # <-- in deg
                    threshold_scale = 20 # <-- in %
                    # <-

                    is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale

                    counter += 1
                    if is_valid_transformation:
                        # GTのBboxを取得
                        t_gt, q_gt, s_gt = SE3_decompose_mat4(Mcad)
                        rotation_gt = quaternion.as_rotation_matrix(q_gt)
                        transration_gt = t_gt
                        scale_gt = s_gt * np.array(model_gt["bbox"]) * 2
                        gt_bbox = Box.from_transformation(rotation_gt, transration_gt, scale_gt)

                        # 推定のBbox
                        est_bbox_pt = Box(estimation_params['bbox_wrd_pt'])
                        est_bbox_dep = Box(estimation_params['bbox_wrd_dep'])

                        # IoUの計算
                        result_iou_pt = IoU(gt_bbox, est_bbox_pt).iou()
                        result_iou_dep = IoU(gt_bbox, est_bbox_dep).iou()

                        # Bboxのデバック
                        import matplotlib.pyplot as plt
                        from mpl_toolkits.mplot3d import Axes3D
                        fig = plt.figure()
                        ax = Axes3D(fig)
                        point1 = gt_bbox._vertices
                        ax.scatter(point1[:, 0], point1[:, 1], point1[:, 2], marker="o", linestyle='None', c='b', s=5)
                        point2 = est_bbox_pt._vertices
                        ax.scatter(point2[:, 0], point2[:, 1], point2[:, 2], marker="o", linestyle='None', c='m', s=5)
                        point3 = est_bbox_dep._vertices
                        ax.scatter(point3[:, 0], point3[:, 1], point3[:, 2], marker="o", linestyle='None', c='r', s=5)
                        ax.view_init(elev=0, azim=90)
                        fig.savefig(f"bbox_{counter}_90.png")
                        ax.view_init(elev=0, azim=0)
                        fig.savefig(f"bbox_{counter}_00.png")
                        ax.view_init(elev=45, azim=45)
                        fig.savefig(f"bbox_{counter}_45.png")
                        plt.close()
                        
                        # 形状のフィット率
                        err_dict_path = ('/').join(alignment.split('/')[:-3] + ['error'] + alignment.split('/')[-2:])
                        with open(err_dict_path, mode='rb') as f:
                            err_dict = pickle.load(f)
                        err_dict['gt_s_fit']
                        err_dict['est_s_fit']

                        # 結果のカウント
                        benchmark_per_scan[id_scan]["n_good"] += 1
                        benchmark_per_class[catname_cad]["n_good"] += 1
                        del groundtruth[key][idx]
                        break

    print("***********")
    benchmark_per_scan = sorted(benchmark_per_scan.items(), key=lambda x: x[1]["n_good"], reverse=True)
    total_accuracy = {"n_good" : 0, "n_total" : 0, "n_scans" : 0}
    for k, v in benchmark_per_scan:
        if "seen" in v:
            total_accuracy["n_good"] += v["n_good"]
            total_accuracy["n_total"] += v["n_total"]
            total_accuracy["n_scans"] += 1
            print("id-scan: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}".format(k,  v["n_good"], v["n_total"], float(v["n_good"])/v["n_total"]))
    instance_mean_accuracy = float(total_accuracy["n_good"])/total_accuracy["n_total"]
    print("instance-mean-accuracy: {:>4.4f} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t n-total-scans: {:>4d}".format(instance_mean_accuracy, total_accuracy["n_good"], total_accuracy["n_total"], total_accuracy["n_scans"]))

    print("*********** PER CLASS **************************")

    accuracy_per_class = {}
    for k,v in benchmark_per_class.items():
        accuracy_per_class[k] = float(v["n_good"])/v["n_total"]
        print("category-name: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}".format(k,  v["n_good"], v["n_total"], float(v["n_good"])/v["n_total"]))

    class_mean_accuracy = np.mean([ v for k,v in accuracy_per_class.items()])
    print("class-mean-accuracy: {:>4.4f}".format(class_mean_accuracy))
    return instance_mean_accuracy, class_mean_accuracy


if __name__ == "__main__":
    evaluate(opt.projectdir, "./data2/scan2cad/cad_appearances.json", "./data2/scan2cad/full_annotations.json")
