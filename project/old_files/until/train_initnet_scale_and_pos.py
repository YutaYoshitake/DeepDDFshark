import os
import pdb
import sys
from turtle import pd
import numpy as np
import random
import pylab
import glob
import math
import re
from tqdm import tqdm, trange
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.autograd import grad
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R

from ResNet import *
from parser import *
from dataset import *
from often_use import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if device=='cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





class TaR_init_only(pl.LightningModule):

    def __init__(self, args, ddf):
        super().__init__()

        # Base configs
        self.dynamic = args.dynamic
        self.fov = args.fov
        self.input_H = 256
        self.input_W = 256
        self.x_coord = torch.arange(0, self.input_W)[None, :].expand(self.input_H, -1)
        self.y_coord = torch.arange(0, self.input_H)[:, None].expand(-1, self.input_W)
        self.image_coord = torch.stack([self.x_coord.T, self.x_coord], dim=-1) # [H, W, (Y and X)]
        self.ddf_H = 256

        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.ddf_H, self.fov)
        self.ddf_instance_list = []
        with open(args.ddf_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.ddf_instance_list.append(line.rstrip('\n'))
        self.save_interval = args.save_interval
        self.optim_step_num = 3 # 5
        self.frame_num = args.frame_num
        self.model_params_dtype = False
        self.model_device = False

        # Make model
        self.ddf = ddf
        self.model = resnet_encoder(args)

        # loss func.
        self.cossim = nn.CosineSimilarity(dim=-1)



    def training_step(self, batch, batch_idx):
        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        batch_size = len(instance_id)

        # Get ground truth.
        instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

        # Set frame.
        frame_idx = random.randint(0, frame_mask.shape[1]-1) # ランダムなフレームを選択

        with torch.no_grad():
            # Clop distance map.
            raw_mask = frame_mask[:, frame_idx]
            raw_distance_map = frame_distance_map[:, frame_idx]
            clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                raw_mask, raw_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H
                                                                )

            # Get normalized depth map.
            rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
            clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                        clopped_mask, clopped_distance_map, rays_d_cam
                                                                        )

            # Get ground truth.
            o2w = frame_obj_rot[:, frame_idx]
            w2c = frame_camera_rot[:, frame_idx]
            o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
            gt_axis_green = o2c[:, :, 1] # Y
            gt_axis_red = o2c[:, :, 0] # X
            cam_pos_wrd = frame_camera_pos[:, frame_idx]
            gt_obj_pos_wrd = frame_obj_pos[:, frame_idx]
            gt_obj_scale = frame_obj_scale[:, frame_idx]
        

        # Get input.
        inp = torch.stack([normalized_depth_map, clopped_mask], 1)
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1).to(inp)
        # axis_info = torch.cat([gt_axis_green, gt_axis_red], dim=-1).to(inp)

        # Estimating.
        # est_x_cim, est_y_cim : クロップされた画像座標（[-1, 1]で定義）における物体中心の予測, 
        # est_z_diff : デプス画像の正則に用いた平均から、物体中心がどれだけズレているか？, 
        # est_axis_green : カメラ座標系での物体の上方向, 
        # est_axis_red : カメラ座標系での物体の右方向, 
        # est_scale_diff : Clopping-BBoxの対角と物体のカノニカルBBoxの対角がどれくらいずれているか, 
        est_x_cim, est_y_cim, est_z_diff, est_axis_green, est_axis_red, est_scale_diff, est_shape_code, _ = self.model(inp, bbox_info) # self.model(inp, bbox_info, axis_info)
        est_obj_pos_cam, est_obj_scale, im2cam_scale = diff2estimation(est_x_cim, est_y_cim, est_z_diff, est_scale_diff, bbox_list, avg_depth_map, self.fov)
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd


        # Cal loss.
        loss_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd)
        loss_scale = F.mse_loss(est_obj_scale, gt_obj_scale.to(est_obj_scale))
        loss_axis_green = torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.)
        loss_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)
        loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code)

        # Cal total loss.
        loss = 1e1 * loss_pos + 1e1 * loss_scale + loss_axis_green + loss_axis_red + 1e1 * loss_shape_code

        # Check distance map.
        if (self.current_epoch+1)//10==0 and batch_idx==0:
            with torch.no_grad():
                rays_d_cam = self.rays_d_cam.expand(2, -1, -1, -1).to(frame_camera_rot.device)
                est_invdistance_map, est_mask, est_distance_map = render_distance_map_from_axis(
                                                                        H = self.ddf_H, 
                                                                        obj_pos_wrd = est_obj_pos_wrd[:2], 
                                                                        axis_green = est_axis_green[:2], 
                                                                        axis_red = est_axis_red[:2], 
                                                                        obj_scale = est_obj_scale[:2], 
                                                                        cam_pos_wrd = cam_pos_wrd[:2], 
                                                                        rays_d_cam = rays_d_cam,  
                                                                        w2c = w2c[:2].detach(), 
                                                                        input_lat_vec = est_shape_code[:2], 
                                                                        ddf = self.ddf, 
                                                                        with_invdistance_map = True, 
                                                                        )
                clopped_est_mask, clopped_est_distance_map, _ = clopping_distance_map(
                                                                    est_mask, est_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H, bbox_list[:2]
                                                                    )

                # Plotを作成
                gt_obj_pos_cam = torch.sum((gt_obj_pos_wrd-cam_pos_wrd)[..., None, :]*w2c, dim=-1)
                fig = pylab.figure(figsize=(20, 8))
                # BBoxをピクセル座標へ
                bbox_list = 128 * (bbox_list.to('cpu').detach().numpy().copy() + 1)
                bbox_center = bbox_list.mean(1)
                obj_pos_cam = 128 * (gt_obj_pos_cam / im2cam_scale[:, None] + 1).to('cpu').detach().numpy().copy()
                obj_pos_cam_ = 128 * (est_obj_pos_cam / im2cam_scale[:, None] + 1).to('cpu').detach().numpy().copy()
                # bbox_center = 128 * (bbox_center.to('cpu').detach().numpy().copy() + 1)
                bbox = np.concatenate([bbox_list, bbox_center[:, None, :], obj_pos_cam[:, None, :2], obj_pos_cam_[:, None, :2]], axis=1)
                # bbox = np.concatenate([bbox_list, obj_pos_cam[:, None, :2]], axis=1)
                bbox_1 = bbox[0]
                bbox_2 = bbox[1]
                # 元画像
                ax_1 = fig.add_subplot(2, 5, 1)
                ax_1.scatter(bbox_1[:, 0], bbox_1[:, 1], c='red', s=20)
                ax_1.imshow(raw_distance_map[0].to('cpu').detach().numpy().copy())
                ax_2 = fig.add_subplot(2, 5, 6)
                ax_2.scatter(bbox_2[:, 0], bbox_2[:, 1], c='red', s=20)
                ax_2.imshow(raw_distance_map[1].to('cpu').detach().numpy().copy())
                # クロップした観測画像
                ax_3 = fig.add_subplot(2, 5, 2)
                ax_3.imshow(clopped_distance_map[0].to('cpu').detach().numpy().copy())
                ax_4 = fig.add_subplot(2, 5, 7)
                ax_4.imshow(clopped_distance_map[1].to('cpu').detach().numpy().copy())
                # 元画像の予測
                ax_5 = fig.add_subplot(2, 5, 3)
                ax_5.scatter(bbox_1[:, 0], bbox_1[:, 1], c='red', s=20)
                ax_5.imshow(est_distance_map[0].to('cpu').detach().numpy().copy())
                ax_6 = fig.add_subplot(2, 5, 8)
                ax_6.scatter(bbox_2[:, 0], bbox_2[:, 1], c='red', s=20)
                ax_6.imshow(est_distance_map[1].to('cpu').detach().numpy().copy())
                # クロップした画像の予測
                ax_7 = fig.add_subplot(2, 5, 4)
                ax_7.imshow(clopped_est_distance_map[0].to('cpu').detach().numpy().copy())
                ax_8 = fig.add_subplot(2, 5, 9)
                ax_8.imshow(clopped_est_distance_map[1].to('cpu').detach().numpy().copy())
                # 誤差
                clopped_error = torch.abs(clopped_distance_map[:2] - clopped_est_distance_map)
                ax_9 = fig.add_subplot(2, 5, 5)
                ax_9.imshow(clopped_error[0].to('cpu').detach().numpy().copy())
                ax_10 = fig.add_subplot(2, 5, 10)
                ax_10.imshow(clopped_error[1].to('cpu').detach().numpy().copy())
                # 画像を保存
                fig.savefig(f"sample_images/initnet_first_test/epo_{str(self.current_epoch).zfill(5)}.png", dpi=300)
                # fig.savefig(f"tes.png", dpi=300)
                pylab.close()

        return {'loss': loss, 'loss_pos':loss_pos.detach(), 'loss_scale': loss_scale.detach(), 'loss_axis_red': loss_axis_red.detach(), 'loss_shape_code': loss_shape_code.detach()}



    def training_epoch_end(self, outputs):

        # Log loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/total_loss': avg_loss, "step": current_epoch})

        avg_loss_pos = torch.stack([x['loss_pos'] for x in outputs]).mean()
        self.log_dict({'train/loss_pos': avg_loss_pos, "step": current_epoch})

        avg_loss_scale = torch.stack([x['loss_scale'] for x in outputs]).mean()
        self.log_dict({'train/loss_scale': avg_loss_scale, "step": current_epoch})

        avg_loss_axis_red = torch.stack([x['loss_axis_red'] for x in outputs]).mean()
        self.log_dict({'train/loss_axis_red': avg_loss_axis_red, "step": current_epoch})

        avg_loss_shape_code = torch.stack([x['loss_shape_code'] for x in outputs]).mean()
        self.log_dict({'train/loss_shape_code': avg_loss_shape_code, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def validation_step(self, batch, batch_idx):
        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        batch_size = len(instance_id)

        # Set frame.
        frame_idx = random.randint(0, frame_mask.shape[1]-1) # ランダムなフレームを選択

        with torch.no_grad():
            # Clop distance map.
            raw_mask = frame_mask[:, frame_idx]
            raw_distance_map = frame_distance_map[:, frame_idx]
            clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                raw_mask, raw_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H
                                                                )

            # Get normalized depth map.
            rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
            clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                        clopped_mask, clopped_distance_map, rays_d_cam
                                                                        )

            # Get ground truth.
            o2w = frame_obj_rot[:, frame_idx]
            w2c = frame_camera_rot[:, frame_idx]
            o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
            gt_axis_green = o2c[:, :, 1] # Y
            gt_axis_red = o2c[:, :, 0] # X
            cam_pos_wrd = frame_camera_pos[:, frame_idx]
            gt_obj_pos_wrd = frame_obj_pos[:, frame_idx]
            gt_obj_scale = frame_obj_scale[:, frame_idx]
        

        # Get input.
        inp = torch.stack([normalized_depth_map, clopped_mask], 1)
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1).to(inp)
        # axis_info = torch.cat([gt_axis_green, gt_axis_red], dim=-1).to(inp)

        # Estimating.
        # est_x_cim, est_y_cim : クロップされた画像座標（[-1, 1]で定義）における物体中心の予測, 
        # est_z_diff : デプス画像の正則に用いた平均から、物体中心がどれだけズレているか？, 
        # est_axis_green : カメラ座標系での物体の上方向, 
        # est_axis_red : カメラ座標系での物体の右方向, 
        # est_scale_diff : Clopping-BBoxの対角と物体のカノニカルBBoxの対角がどれくらいずれているか, 
        est_x_cim, est_y_cim, est_z_diff, est_axis_green, est_axis_red, est_scale_diff, est_shape_code, _ = self.model(inp, bbox_info) # self.model(inp, bbox_info, axis_info)
        est_obj_pos_cam, est_obj_scale, im2cam_scale = diff2estimation(est_x_cim, est_y_cim, est_z_diff, est_scale_diff, bbox_list, avg_depth_map, self.fov)
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd


        # Cal err.
        err_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd)
        err_scale = torch.mean(est_obj_scale / gt_obj_scale.to(est_obj_scale))
        err_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)

        return {'err_pos':err_pos.detach(), 'err_scale': err_scale.detach(), 'err_axis_red': err_axis_red.detach()}



    def validation_epoch_end(self, outputs):
        # Log err.
        avg_err_axis_red = torch.stack([x['err_axis_red'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_err_axis_red.dtype)
        self.log_dict({'validation/err_axis_red': avg_err_axis_red, "step": current_epoch})

        avg_err_scale = torch.stack([x['err_scale'] for x in outputs]).mean()
        self.log_dict({'validation/err_scale': avg_err_scale, "step": current_epoch})

        avg_err_pos = torch.stack([x['err_pos'] for x in outputs]).mean()
        self.log_dict({'validation/err_pos': avg_err_pos, "step": current_epoch})




    def check_model_info (self):
        self.model_params_dtype = list(self.mlp.parameters())[-1].dtype
        self.model_device = self.device



    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
        ], lr=self.lr, betas=(0.9, 0.999),)
        return optimizer



if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.


    # Set trainer.
    logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f'{args.expname}_{args.exp_version}',
            name='lightning_logs'
        )
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=True), #=False), 
        logger=logger,
        max_epochs=args.N_epoch, 
        enable_checkpointing = False,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        )
    

    # Save config files.
    os.makedirs(os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}'), exist_ok=True)
    f = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create dataloader
    train_dataset = TaR_dataset(
        args, 
        'train', 
        args.train_instance_list_txt, 
        args.train_data_dir, 
        args.train_N_views, 
        )
    train_dataloader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=True, 
        )
    val_dataset = TaR_dataset(
        args, 
        'val', 
        args.val_instance_list_txt, 
        args.val_data_dir, 
        args.val_N_views, 
        )
    val_dataloader = data_utils.DataLoader(
        val_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=False, 
        )

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    
    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    # if len(ckpt_path_list) == 0:
    model = TaR_init_only(args, ddf)
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        datamodule=None, 
        ckpt_path=None
        )

    # elif len(ckpt_path_list) > 0:
    #     latest_ckpt_path = ckpt_path_list[-1]
    #     print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')
    #     model = TaR_init_only(args, ddf)
    #     trainer.fit(
    #         model=model, 
    #         train_dataloaders=train_dataloader, 
    #         val_dataloaders=val_dataloader, 
    #         datamodule=None, 
    #         ckpt_path=latest_ckpt_path
    #         )
