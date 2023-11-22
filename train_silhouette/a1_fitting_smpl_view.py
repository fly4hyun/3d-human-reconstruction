




import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)



import mediapipe as mp
import glob
from termcolor import colored
from tqdm.auto import tqdm
import os
from yacs.config import CfgNode as CN
from PIL import Image

import torch
from torch import nn
import numpy as np
import trimesh
import argparse

from a0_train_data_loader import TestDataset
from models_LVD_encoder.LVD_encoder import Network
import models_SMPL_ICON.smplx as smplx
from models_SMPL_ICON.mesh_util import SMPLX, remesh, rot6d_to_rotmat
from models_SMPL_LVD.SMPL import SMPL

from models_SMPL_LVD.prior import SMPLifyAnglePrior, MaxMixturePrior
from models_SMPL_LVD.util_smpl import batch_rodrigues


from renderlib.render import Render, query_color, query_color_4v
from renderlib.mesh import compute_normal_batch
from models_ICON_IMGGAN.NormalNet import NormalNet
from models_ICON_IMGGAN.ICON import ICON
from model_save.config import cfg

from pytorch3d.structures import Meshes

import pywavefront
from body_measurements.measurement import Body3D
from copy import copy
from models_seg import networks

##################################################
##### parameter setting #####
##################################################

def loss_smpl(a, b):
    a = torch.unsqueeze(a, dim=1)
    b = torch.unsqueeze(b, dim=2)
    c = torch.pow(torch.sub(a, b), 2)
    c = torch.pow(torch.sum(c, dim=-1), 1/2)

    a_loss = torch.sum(torch.amax(c, 1)) / 2
    b_loss = torch.sum(torch.amax(c, 2)) /2
    
    return a_loss + b_loss







parser = argparse.ArgumentParser()

parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
parser.add_argument("-colab", action="store_true")
parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=100)
parser.add_argument("-patience", "--patience", type=int, default=5)
parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=200)
parser.add_argument("-hps_type", "--hps_type", type=str, default="pymaf")
parser.add_argument("-export_video", action="store_true")
parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
parser.add_argument('-seg_dir', '--seg_dir', type=str, default=None)
parser.add_argument("-cfg",
                    "--config",
                    type=str,
                    default="./model_save/ICON_img/icon-filter.yaml")

args = parser.parse_args()

# cfg read and merge
cfg.merge_from_file(args.config)
cfg.merge_from_file("./models_pymaf/pymaf_config.yaml")

cfg_show_list = [
    "test_gpus", [args.gpu_device], "mcube_res", 256, "clean_mesh", True,
    "test_mode", True, "batch_size", 1
]

cfg.merge_from_list(cfg_show_list)
cfg.freeze()

















##### SMPL #####

smpl_type = 'smpl'
smpl_gender = 'female'
# male
# female
# neutral

##### LVD Encoder #####

LVD_loop = 10000
LVD_lr = 1e-2


loop_cloth__ = 500



##### device setting #####

device = torch.device('cuda:0')

##################################################
##### image import #####
##################################################

render = Render(size = 512, device = device)
dataset = TestDataset(device)







##################################################
##### model import #####
##################################################

body_seg_model = networks.init_model('resnet101', num_classes=7, pretrained=None)
state_dict = torch.load('models_seg/exp-schp-201908270938-pascal-person-part.pth')['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
body_seg_model.load_state_dict(new_state_dict)
body_seg_model.to(device)
body_seg_model.eval()

##### SMPL import #####

##### SMPL import (ICON) #####


smpl_data = SMPLX()
get_smpl_model = lambda smpl_type, smpl_gender: smplx.create(
    model_path = smpl_data.model_dir,
    gender = smpl_gender,
    model_type = smpl_type,
    ext = 'npz'
    )

SMPL = get_smpl_model(smpl_type, smpl_gender).to(device)
faces = SMPL.faces




pbar = tqdm(dataset)

for data in pbar:
    
    class OptimizationSMPL(torch.nn.Module):
        def __init__(self):
            super(OptimizationSMPL, self).__init__()

            #self.orient = torch.nn.Parameter(torch.randn([1, 1], dtype=torch.float32).to(device))
            #self.pose = torch.nn.Parameter(torch.randn([1, 23], dtype=torch.float32).to(device))
            #self.betas = torch.nn.Parameter(torch.randn([1, 300], dtype=torch.float32).to(device))
            #self.trans = torch.nn.Parameter(torch.randn([1, 3], dtype=torch.float32).to(device))
            #self.scale = torch.nn.Parameter(torch.ones([1], dtype=torch.float32).to(device)) 
            self.orient = torch.nn.Parameter(data['orient']).to(device)
            self.pose = torch.nn.Parameter(data['pose']).to(device)
            self.betas = torch.nn.Parameter(data['beta']).to(device)
            self.trans = torch.nn.Parameter(data['trans']).to(device)
            self.scale = torch.nn.Parameter(data['scale']).to(device)
            
            
            
        def forward(self):
            # return self.orient, self.pose, self.theta, self.betas, self.expression, self.trans, self.scale
            return self.orient, self.pose, self.betas, self.trans, self.scale
    
    e3 = torch.eye(3).float().to(device)
    parameters_smpl = OptimizationSMPL().to(device)
    optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr = 0.01)
    
    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_smpl,
        mode="min",
        factor=0.5,
        verbose=0,
        min_lr=1e-4,
        patience=20,
    )
    # scheduler_smpl = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer_smpl,
    #     base_lr=0.01, 
    #     step_size_up=5, 
    #     max_lr=0.0001, 
    #     mode = 'triangular2', 
    #     cycle_momentum = False
    # )
    
    print(data['name'])
    
    iterations = tqdm(range(300))
    for i_num in iterations:
        
        # optimed_theta, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
        # optimed_theta, optimed_beta, optimed_expression, optimed_trans, optimed_scale = parameters_smpl.forward()
        optimed_orient, optimed_pose, optimed_beta, optimed_trans, optimed_scale = parameters_smpl.forward()
        
        # Rs = batch_rodrigues(optimed_theta.view(-1, 3)).view(-1, 24, 3, 3)
        # optimed_pose = (Rs[:, 1:, :, :]).sub(1.0, e3)#.view(-1, 207)
        # optimed_orient = Rs[:, :1, :, :]
        
        smpl_out = SMPL(global_orient = optimed_orient, 
                        betas = optimed_beta, 
                        body_pose = optimed_pose, 
                        transl=optimed_trans,
                        pose2rot=False
                        ).vertices * optimed_scale
        # smpl_out = (SMPL(betas=optimed_betas,
        #     body_pose=optimed_pose,
        #     global_orient=optimed_orient,
        #     transl=optimed_trans,
        #     pose2rot=False
        # ).vertices + optimed_trans) * optimed_scale
        
        #smpl_out = 1.8 * ((smpl_out - smpl_out[0, :, 1].min()) / (smpl_out[0, :, 1].max() - smpl_out[0, :, 1].min())) - 0.89
        
        smpl_out = smpl_out * torch.tensor([1.0, -1.0, -1.0]).to(device)
        
        index_1 = torch.tensor([2, 1, 0]).unsqueeze(0).unsqueeze(0)
        index_1 = index_1.expand(smpl_out.size(0), smpl_out.size(1), index_1.size(2))
        index_1 = index_1.to(device)
        
        
        render.load_meshes(smpl_out, torch.tensor(faces.astype(np.int64)))
        smpl_img_0, _ = render.get_rgb_image()
        T_mask_0, _ = render.get_silhouette_image()
        
        
        
        render.load_meshes(smpl_out * torch.tensor([-1.0, 1.0, -1.0]).to(device), torch.tensor(faces.astype(np.int64)))
        smpl_img_2, _ = render.get_rgb_image()
        T_mask_2, _ = render.get_silhouette_image()
        

        
        render.load_meshes(smpl_out.gather(2, index_1) * torch.tensor([-1.0, 1.0, 1.0]).to(device), torch.tensor(faces.astype(np.int64)))
        smpl_img_1, _ = render.get_rgb_image()
        T_mask_1, _ = render.get_silhouette_image()
        
        render.load_meshes(smpl_out.gather(2, index_1) * torch.tensor([1.0, 1.0, -1.0]).to(device), torch.tensor(faces.astype(np.int64)))
        smpl_img_3, _ = render.get_rgb_image()
        T_mask_3, _ = render.get_silhouette_image()
        
        
        
        
        
        #############################asdfasfasdfasdfdf


        
        
        img_0 = ((smpl_img_0[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_0).save('./_0.png')
        img_1 = ((smpl_img_1[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_1).save('./_1.png')
        img_2 = ((smpl_img_2[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_2).save('./_2.png')
        img_3 = ((smpl_img_3[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_3).save('./_3.png')
        
        img_0_0 = ((data['image_0'][0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_0_0).save('./_0_0.png')
        img_1_0 = ((data['image_1'][0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_1_0).save('./_1_0.png')
        img_2_0 = ((data['image_2'][0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_2_0).save('./_2_0.png')
        img_3_0 = ((data['image_3'][0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_3_0).save('./_3_0.png')
        
        img_0_b = ((T_mask_0[0] + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_0_b).save('./_0_b.png')
        img_1_b = ((T_mask_1[0] + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_1_b).save('./_1_b.png')
        img_2_b = ((T_mask_2[0] + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_2_b).save('./_2_b.png')
        img_3_b = ((T_mask_3[0] + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(img_3_b).save('./_3_b.png')
        
        
        # normal loss
        diff_loss_0 = torch.abs(smpl_img_0[0] - data['image_0'][0].to(device))
        diff_loss_1 = torch.abs(smpl_img_1[0] - data['image_1'][0].to(device))
        diff_loss_2 = torch.abs(smpl_img_2[0] - data['image_2'][0].to(device))
        diff_loss_3 = torch.abs(smpl_img_3[0] - data['image_3'][0].to(device))

        
        normal_loss = (diff_loss_0 + diff_loss_1 + diff_loss_2 + diff_loss_3).mean()
        # normal_loss = (diff_loss + diff_loss).mean()
        
        # silhouette loss
        smpl_arr = torch.cat([T_mask_0, T_mask_0, T_mask_1, T_mask_2, T_mask_2, T_mask_3], dim=-1)[0]
        gt_arr = torch.cat([data['image_0'][0], data['image_0'][0], data['image_1'][0], data['image_2'][0], data['image_2'][0], data['image_3'][0]], dim=2).permute(1, 2, 0)
        gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
        bg_color = (torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device))
        gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
        diff_S = torch.abs(smpl_arr - gt_arr)

        gt_arr = (gt_arr * 255.0 / 2.0 / 0.5).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(gt_arr).save('./_0_gt.png')
        smpl_arr = (smpl_arr * 255.0 / 2.0 / 0.5).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(smpl_arr).save('./_0_smpl_arr.png')

        
        silhouette_loss = diff_S.mean()
        
        
        # arm loss
        
        T_arm_mask_1 = body_seg_model(data['image_1'].to(device))
        smpl_arm_mask_1 = body_seg_model(smpl_img_1.to(device))
        
        T_arm_mask_0 = body_seg_model(data['image_0'].to(device))
        smpl_arm_mask_0 = body_seg_model(smpl_img_0.to(device))
        
        T_arm_mask_3 = body_seg_model(data['image_3'].to(device))
        smpl_arm_mask_3 = body_seg_model(smpl_img_3.to(device))
        
        T_arm_mask_2 = body_seg_model(data['image_2'].to(device))
        smpl_arm_mask_2 = body_seg_model(smpl_img_2.to(device))
        
        
        T_arm_mask_1 = T_arm_mask_1[0][-1][0].unsqueeze(0)
        T_arm_mask_1 = T_arm_mask_1.squeeze()
        T_arm_mask_1 = T_arm_mask_1.permute(1, 2, 0)
        T_arm_mask_1 = T_arm_mask_1.argmax(dim = 2)
        
        smpl_arm_mask_1 = smpl_arm_mask_1[0][-1][0].unsqueeze(0)
        smpl_arm_mask_1 = smpl_arm_mask_1.squeeze()
        smpl_arm_mask_1 = smpl_arm_mask_1.permute(1, 2, 0)
        smpl_arm_mask_1 = smpl_arm_mask_1.argmax(dim = 2)
        
        Image.fromarray((T_arm_mask_1 * 255 / 8).data.cpu().numpy().astype(np.uint8)).save('./_0_seg_0.png')
        Image.fromarray((smpl_arm_mask_1 * 255 / 8).data.cpu().numpy().astype(np.uint8)).save('./_0_seg_1.png')
        
        T_arm_mask_0 = T_arm_mask_0[0][-1][0].unsqueeze(0)
        T_arm_mask_0 = T_arm_mask_0.squeeze()
        T_arm_mask_0 = T_arm_mask_0.permute(1, 2, 0)
        T_arm_mask_0 = T_arm_mask_0.argmax(dim = 2)
        
        smpl_arm_mask_0 = smpl_arm_mask_0[0][-1][0].unsqueeze(0)
        smpl_arm_mask_0 = smpl_arm_mask_0.squeeze()
        smpl_arm_mask_0 = smpl_arm_mask_0.permute(1, 2, 0)
        smpl_arm_mask_0 = smpl_arm_mask_0.argmax(dim = 2)
        
        T_arm_mask_3 = T_arm_mask_3[0][-1][0].unsqueeze(0)
        T_arm_mask_3 = T_arm_mask_3.squeeze()
        T_arm_mask_3 = T_arm_mask_3.permute(1, 2, 0)
        T_arm_mask_3 = T_arm_mask_3.argmax(dim = 2)
        
        smpl_arm_mask_3 = smpl_arm_mask_3[0][-1][0].unsqueeze(0)
        smpl_arm_mask_3 = smpl_arm_mask_3.squeeze()
        smpl_arm_mask_3 = smpl_arm_mask_3.permute(1, 2, 0)
        smpl_arm_mask_3 = smpl_arm_mask_3.argmax(dim = 2)
        
        T_arm_mask_2 = T_arm_mask_2[0][-1][0].unsqueeze(0)
        T_arm_mask_2 = T_arm_mask_2.squeeze()
        T_arm_mask_2 = T_arm_mask_2.permute(1, 2, 0)
        T_arm_mask_2 = T_arm_mask_2.argmax(dim = 2)
        
        smpl_arm_mask_2 = smpl_arm_mask_2[0][-1][0].unsqueeze(0)
        smpl_arm_mask_2 = smpl_arm_mask_2.squeeze()
        smpl_arm_mask_2 = smpl_arm_mask_2.permute(1, 2, 0)
        smpl_arm_mask_2 = smpl_arm_mask_2.argmax(dim = 2)
        
        
        
        
        
        arm_loss_1 = torch.abs(T_arm_mask_1.float() - smpl_arm_mask_1.float())
        arm_loss_0 = torch.abs(T_arm_mask_0.float() - smpl_arm_mask_0.float())
        arm_loss_3 = torch.abs(T_arm_mask_3.float() - smpl_arm_mask_3.float())
        arm_loss_2 = torch.abs(T_arm_mask_2.float() - smpl_arm_mask_2.float())
        
        arm_loss = (arm_loss_1 + arm_loss_3).mean()

        loss = 1 * normal_loss + 1 * silhouette_loss + 1 * arm_loss#loss_smpl(smpl_out.to(device), data['verts'].to(device))
        
        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()
        scheduler_smpl.step(loss)
        
        pbar_desc = "SMPL fitting --- "
        pbar_desc += f"Total loss: {loss.item():.5f}"
        iterations.set_description(pbar_desc)

        
    #optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).unsqueeze(0)
    #optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).unsqueeze(0)
    with torch.no_grad():
        smpl_out = SMPL(global_orient = optimed_orient, 
                        betas = optimed_beta, 
                        body_pose = optimed_pose, 
                        transl=optimed_trans,
                        pose2rot=False
                        ).vertices * optimed_scale


        smpl_out = smpl_out * torch.tensor([1.0, -1.0, -1.0]).to(device)

        m = trimesh.Trimesh(smpl_out[0].cpu().detach(), faces, process = False, maintains_order = True)
        m.export('./test.obj')
    
    
    
    asdfaf
    
    