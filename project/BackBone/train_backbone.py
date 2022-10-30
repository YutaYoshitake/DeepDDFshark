import os
import sys
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
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append("../")
from parser_get_arg import *
from often_use import *





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False) # nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.relu_inplace = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True)
        self.downsample = downsample
        if BatchNorm is None:
            print('##### No Norms #####')
            self.bn1 = None
        else:
            self.bn1 = BatchNorm(planes)
            self.bn2 = BatchNorm(planes)

    def forward(self, x):
        residual = x
        if self.bn1 is None:
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual      
        out = self.relu_inplace(out)
        return out



class encoder_2dcnn(nn.Module):

    def __init__(self, in_channel=4):
        self.inplanes = 64
        super(encoder_2dcnn, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        BatchNorm = nn.BatchNorm2d
        blocks = [1, 2, 4]
        self.each_layers_num = layers
        self.embedding_dim = 512

        # Modules
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.bn1 = BatchNorm(64)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1, dilation=1, BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=1, BatchNorm=BatchNorm)
        self.layer5 = nn.AdaptiveAvgPool2d((1,1))
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def fix_norms(self):
        # layer1.
        for layer_idx in range(self.each_layers_num[0]):
            self.layer1[layer_idx].bn1 = fixed_BatchNorm2d(self.layer1[layer_idx].bn1)
            self.layer1[layer_idx].bn2 = fixed_BatchNorm2d(self.layer1[layer_idx].bn2)
            if self.layer1[layer_idx].downsample is not None: 
                self.layer1[layer_idx].downsample[1] = \
                    fixed_BatchNorm2d(self.layer1[layer_idx].downsample[1])
        # layer2.
        for layer_idx in range(self.each_layers_num[1]):
            self.layer2[layer_idx].bn1 = fixed_BatchNorm2d(self.layer2[layer_idx].bn1)
            self.layer2[layer_idx].bn2 = fixed_BatchNorm2d(self.layer2[layer_idx].bn2)
            if self.layer2[layer_idx].downsample is not None: 
                self.layer2[layer_idx].downsample[1] = \
                    fixed_BatchNorm2d(self.layer2[layer_idx].downsample[1])
        # layer3.
        for layer_idx in range(self.each_layers_num[2]):
            self.layer3[layer_idx].bn1 = fixed_BatchNorm2d(self.layer3[layer_idx].bn1)
            self.layer3[layer_idx].bn2 = fixed_BatchNorm2d(self.layer3[layer_idx].bn2)
            if self.layer3[layer_idx].downsample is not None: 
                self.layer3[layer_idx].downsample[1] = \
                    fixed_BatchNorm2d(self.layer3[layer_idx].downsample[1])
        # layer4.
        for layer_idx in range(self.each_layers_num[3]):
            self.layer4[layer_idx].bn1 = fixed_BatchNorm2d(self.layer4[layer_idx].bn1)
            self.layer4[layer_idx].bn2 = fixed_BatchNorm2d(self.layer4[layer_idx].bn2)
            if self.layer4[layer_idx].downsample is not None: 
                self.layer4[layer_idx].downsample[1] = \
                    fixed_BatchNorm2d(self.layer4[layer_idx].downsample[1])
        # Check norms in all layers
        self.bn1 = fixed_BatchNorm2d(self.bn1)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                print('Failed to fix all norms!')
                sys.exit()



class encoder_2dcnn_wonorms(nn.Module):

    def __init__(self, in_channel=4):
        self.inplanes = 64
        super(encoder_2dcnn_wonorms, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        blocks = [1, 2, 4]
        self.each_layers_num = layers
        self.embedding_dim = 512

        # Modules
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1, dilation=1, BatchNorm=None)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1, BatchNorm=None)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, BatchNorm=None)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=1, BatchNorm=None)
        self.layer5 = nn.AdaptiveAvgPool2d((1,1))
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class fixed_BatchNorm2d(nn.Module):

    def __init__(self, BatchNorm):
        super(fixed_BatchNorm2d, self).__init__()
        if not isinstance(BatchNorm, nn.BatchNorm2d):
            print('Input error!')
            sys.exit()
        # Get each params.
        # [dummy_batch, C, dummy_H, dummy_W]
        self.bias = BatchNorm.bias.data[None, :, None, None]
        self.weight = BatchNorm.weight.data[None, :, None, None]
        self.mean = BatchNorm.running_mean[None, :, None, None]
        self.var = BatchNorm.running_var[None, :, None, None]
        self.eps = BatchNorm.eps

    def forward(self, x):
        x = (x - self.mean) / torch.sqrt(self.var + self.eps) * self.weight + self.bias
        return x



class decoder_2dcnn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decoder_2dcnn, self).__init__()

        # Modules
        hidden_dim = 1024
        self.layer1 = nn.Conv2d(in_channel, 2*hidden_dim, 1)
        self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(2*hidden_dim, hidden_dim//2, 4, 1, 0),
                nn.BatchNorm2d(hidden_dim//2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )
        self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim//4),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )
        self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim//8),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )
        self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim//16),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )
        self.layer6 = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim//16, hidden_dim//32, 4, 2, 1),
                nn.BatchNorm2d(hidden_dim//32),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )
        self.layer7 = nn.ConvTranspose2d(hidden_dim//32, out_channel, 4, 2, 1)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class backbone_dataset(data.Dataset):
    def __init__(
        self, 
        args, 
        dir_list, 
        N_views, 
        N_randn, 
        instance_list_txt, 
        ):

        self.N_views = N_views
        self.data_list = []
        for dir_i in dir_list:
            with open(instance_list_txt, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    for view_id in range(1, self.N_views+1):
                        str_view_id = str(view_id).zfill(5)
                        path_pickle = os.path.join(dir_i, line.rstrip('\n'), f'{str_view_id}_{str(0).zfill(2)}.pickle')
                        self.data_list.append(path_pickle)
                        if dir_i.split('/')[-1] in {'pre', 'dif'}:
                            for randn_idx in range(1, N_randn):
                                path_pickle = os.path.join(dir_i, line.rstrip('\n'), f'{str_view_id}_{str(randn_idx).zfill(2)}.pickle')
                                self.data_list.append(path_pickle)
        # self.data_list = self.data_list[:2]

    def __getitem__(self, index):

        path = self.data_list[index]
        data_dict = pickle_load(path)
        clopped_mask = torch.from_numpy(data_dict['clopped_mask'].astype(np.float32)).clone()
        clopped_distance_map = torch.from_numpy(data_dict['clopped_distance_map'].astype(np.float32)).clone()
        OSMap_wrd = torch.from_numpy(data_dict['osmap_wrd'].astype(np.float32)).clone()

        return clopped_mask, clopped_distance_map, OSMap_wrd

    def __len__(self):
        return len(self.data_list)



class backbone_encoder_decoder(pl.LightningModule):

    def __init__(self, args):
        super(backbone_encoder_decoder, self).__init__()

        # Configs.
        self.lr = args.lr_backbone
        self.L_reg = 1.e-3
        self.L_recon_map = 1.0
        self.L_recon_mask = 1.e-1
        self.log_image_interval = 1
        self.save_interval = args.save_interval
        if args.gpu_num > 1:
            print('only single gpu')
            sys.exit()
        self.map_H = 128
        self.map_W = 128
        if args.input_type == 'depthmap':
            self.map_channel = 2
        elif args.input_type == 'osmap':
            self.map_channel = 4
        self.embedding_dim = 512

        # Models.
        if args.backbone_norms=='with_out_norm':
            self.encoder_2dcnn = encoder_2dcnn_wonorms(self.map_channel)
        elif args.backbone_norms=='with_norm':
            self.encoder_2dcnn = encoder_2dcnn(self.map_channel)
        self.decoder_2dcnn = decoder_2dcnn(self.embedding_dim, self.map_channel).apply(weights_init)
        self.mask_act = nn.Sigmoid()

        # Loss
        self.BCELoss = nn.BCELoss()
        self.MSELoss = nn.MSELoss()


    def forward(self, inp):
        embedding = self.encoder_2dcnn(inp)
        x = self.decoder_2dcnn(embedding)
        recon_map  = x[:, :-1, :, :] # get map part.
        recon_mask = x[:, -1, :, :] # get mask part.
        recon_mask = self.mask_act(recon_mask)
        return embedding, recon_map, recon_mask


    def training_step(self, batch, batch_idx, step_mode='train'):

        # Get batch.
        gt_clopped_mask = batch[0] # [batch, H, W]
        gt_OSMap_wrd = batch[2].permute(0, 3, 1, 2) # [batch, C, H, W]
        batch_size = gt_clopped_mask.shape[0]

        # Estimating.
        inp = torch.cat([gt_OSMap_wrd, gt_clopped_mask.unsqueeze(dim=1)], dim=1).contiguous().detach()
        embedding, recon_map, recon_mask = self.forward(inp)

        # Cal loss.
        embedding_norm = torch.norm(embedding.squeeze(-1).squeeze(-1), dim=-1)
        loss_reg = torch.sum(embedding_norm, dim=-1) / embedding.shape[0]
        loss_recon_map = self.MSELoss(recon_map, gt_OSMap_wrd)
        loss_recon_mask = self.BCELoss(recon_mask, gt_clopped_mask)
        loss = self.L_recon_map * loss_recon_map\
             + self.L_recon_mask * loss_recon_mask\
             + self.L_reg * min(1, self.current_epoch / 10) * loss_reg

        # Log images.
        if batch_idx == 0:
            log_size = min(batch_size, 5)
            est_mask = recon_mask[:log_size] > 0.5 # [batch, H, W]
            est_map  = recon_map[:log_size] # [batch, C, H, W]
            gt_mask  = gt_clopped_mask[:log_size] > 0.5 # [batch, H, W]
            gt_map   = gt_OSMap_wrd[:log_size]
            self.log_maps(est_mask, est_map, gt_mask, gt_map, step_mode)
        
        # Return values.
        return {"loss": loss, 
                "loss_reg": loss_reg.detach(), 
                "loss_map": loss_recon_map.detach(), 
                "loss_mask": loss_recon_mask.detach(), }


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/loss': avg_loss, "step": current_epoch})
        avg_loss_reg = torch.stack([x['loss_reg'] for x in outputs]).mean()
        self.log_dict({'train/loss_reg': avg_loss_reg, "step": current_epoch})
        avg_loss_map = torch.stack([x['loss_map'] for x in outputs]).mean()
        self.log_dict({'train/loss_map': avg_loss_map, "step": current_epoch})
        avg_loss_mask = torch.stack([x['loss_mask'] for x in outputs]).mean()
        self.log_dict({'train/loss_mask': avg_loss_mask, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)


    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, step_mode='val')


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'val/loss': avg_loss, "step": current_epoch})
        avg_loss_reg = torch.stack([x['loss_reg'] for x in outputs]).mean()
        self.log_dict({'val/loss_reg': avg_loss_reg, "step": current_epoch})
        avg_loss_map = torch.stack([x['loss_map'] for x in outputs]).mean()
        self.log_dict({'val/loss_map': avg_loss_map, "step": current_epoch})
        avg_loss_mask = torch.stack([x['loss_mask'] for x in outputs]).mean()
        self.log_dict({'val/loss_mask': avg_loss_mask, "step": current_epoch})


    def log_maps(self, est_mask, est_map, gt_mask, gt_map, code_mode):
            xyz_min = gt_map.permute(0, 2, 3, 1)[gt_mask].min(dim=0)[0]
            xyz_max = gt_map.permute(0, 2, 3, 1)[gt_mask].max(dim=0)[0]
            normalized_gt_map =  (gt_map  - xyz_min[None, :, None, None]) / (xyz_max - xyz_min)[None, :, None, None]
            normalized_est_map = (est_map - xyz_min[None, :, None, None]) / (xyz_max - xyz_min)[None, :, None, None]
            normalized_est_map = torch.clip(normalized_est_map, min=0.0, max=1.0)

            self.logger.experiment.add_image(
                    tag=f'{code_mode}/est_mask',
                    img_tensor=make_grid(est_mask.unsqueeze(1), nrow=8, padding=0),
                    global_step=self.global_step, )
            self.logger.experiment.add_image(
                    tag=f'{code_mode}/est_map',
                    img_tensor=make_grid(normalized_est_map, nrow=8, padding=0),
                    global_step=self.global_step, )
            self.logger.experiment.add_image(
                    tag=f'{code_mode}/gt_mask',
                    img_tensor=make_grid(gt_mask.unsqueeze(1), nrow=8, padding=0),
                    global_step=self.global_step, )
            self.logger.experiment.add_image(
                    tag=f'{code_mode}/gt_map',
                    img_tensor=make_grid(normalized_gt_map, nrow=8, padding=0),
                    global_step=self.global_step, )
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.parameters()},
        ], lr=self.lr, betas=(0.9, 0.999),)
        return optimizer



if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.check_val_every_n_epoch = args.save_interval
    args.backboneconfs_datadir_list = args.backboneconfs_datadir_list.split('_')
    args.train_datadir_list = [os.path.join(args.train_data_dir, dir_i) for dir_i in args.backboneconfs_datadir_list]
    args.val_datadir_list = [os.path.join(args.val_data_dir, dir_i) for dir_i in args.backboneconfs_datadir_list]


    # Create dataloader
    train_dataset = backbone_dataset(
                        args = args, 
                        dir_list = args.train_datadir_list, 
                        N_views = args.train_N_views, 
                        N_randn = args.backboneconfs_N_randn, 
                        instance_list_txt = args.train_instance_list_txt)
    train_dataloader = data_utils.DataLoader(
                        dataset = train_dataset, 
                        batch_size=args.N_batch, 
                        num_workers=args.num_workers, 
                        shuffle=True, 
                        drop_last=True)
    val_dataset = backbone_dataset(
                        args = args, 
                        dir_list = args.val_datadir_list, 
                        N_views = args.val_N_views, 
                        N_randn = args.backboneconfs_N_randn, 
                        instance_list_txt = args.val_instance_list_txt)
    val_dataloader = data_utils.DataLoader(
                        dataset = val_dataset, 
                        batch_size=args.N_batch, 
                        num_workers=args.num_workers, 
                        shuffle=True, 
                        drop_last=True)


    # Set trainer.
    logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f'{args.expname}_{args.exp_version}',
            name='lightning_logs'
        )
    
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=False), # =True), 
        logger=logger,
        max_epochs=args.N_epoch, 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        )
    
    model = backbone_encoder_decoder(args)
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        ckpt_path=None, 
        val_dataloaders=val_dataloader, 
        datamodule=None, 
        )
