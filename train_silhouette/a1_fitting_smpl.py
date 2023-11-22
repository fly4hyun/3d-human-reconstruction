




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
        factor=0.8,
        verbose=0,
        min_lr=1e-5,
        patience=5,
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
    
    iterations = tqdm(range(400))
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

        
        
        # normal loss
        diff_loss_0 = torch.abs(smpl_img_0[0] - data['image_0'][0].to(device))
        diff_loss_1 = torch.abs(smpl_img_1[0] - data['image_1'][0].to(device))
        diff_loss_2 = torch.abs(smpl_img_2[0] - data['image_2'][0].to(device))
        diff_loss_3 = torch.abs(smpl_img_3[0] - data['image_3'][0].to(device))

        
        normal_loss = (diff_loss_0 + diff_loss_1 + diff_loss_2 + diff_loss_3).mean()
        # normal_loss = (diff_loss + diff_loss).mean()
        
        # silhouette loss
        smpl_arr = torch.cat([T_mask_0, T_mask_1, T_mask_2, T_mask_3], dim=-1)[0]
        gt_arr = torch.cat([data['image_0'][0], data['image_1'][0], data['image_2'][0], data['image_3'][0]], dim=2).permute(1, 2, 0)
        gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
        bg_color = (torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device))
        gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
        diff_S = torch.abs(smpl_arr - gt_arr)


        
        silhouette_loss = diff_S.mean()
        
        
        # arm loss
        
        T_arm_mask_1 = body_seg_model(data['image_1'].to(device))
        smpl_arm_mask_1 = body_seg_model(smpl_img_1.to(device))

        
        T_arm_mask_3 = body_seg_model(data['image_3'].to(device))
        smpl_arm_mask_3 = body_seg_model(smpl_img_3.to(device))
        

        T_arm_mask_1 = T_arm_mask_1[0][-1][0].unsqueeze(0)
        T_arm_mask_1 = T_arm_mask_1.squeeze()
        T_arm_mask_1 = T_arm_mask_1.permute(1, 2, 0)
        T_arm_mask_1 = T_arm_mask_1.argmax(dim = 2)
        
        smpl_arm_mask_1 = smpl_arm_mask_1[0][-1][0].unsqueeze(0)
        smpl_arm_mask_1 = smpl_arm_mask_1.squeeze()
        smpl_arm_mask_1 = smpl_arm_mask_1.permute(1, 2, 0)
        smpl_arm_mask_1 = smpl_arm_mask_1.argmax(dim = 2)


        
        T_arm_mask_3 = T_arm_mask_3[0][-1][0].unsqueeze(0)
        T_arm_mask_3 = T_arm_mask_3.squeeze()
        T_arm_mask_3 = T_arm_mask_3.permute(1, 2, 0)
        T_arm_mask_3 = T_arm_mask_3.argmax(dim = 2)
        
        smpl_arm_mask_3 = smpl_arm_mask_3[0][-1][0].unsqueeze(0)
        smpl_arm_mask_3 = smpl_arm_mask_3.squeeze()
        smpl_arm_mask_3 = smpl_arm_mask_3.permute(1, 2, 0)
        smpl_arm_mask_3 = smpl_arm_mask_3.argmax(dim = 2)
        

        
        
        
        arm_loss_1 = torch.abs(T_arm_mask_1.float() - smpl_arm_mask_1.float())

        arm_loss_3 = torch.abs(T_arm_mask_3.float() - smpl_arm_mask_3.float())

        
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
        m.export('./woman/obj_smpl/%s.obj'%data['name'])
    


asfasdfasf







































asdfasdfasfd


##### LVD Encoder import #####


cfg.merge_from_file('./model_save/ICON_img/icon-filter.yaml')
cfg_show_list = [
    "test_gpus", [0], "mcube_res", 256, "clean_mesh", True,
    "test_mode", True, "batch_size", 1
]
cfg.merge_from_list(cfg_show_list)
cfg.freeze()

cfg.merge_from_file('./model_save/ICON_img/pymaf_config.yaml')

model = ICON(cfg)
from models_SMPL_ICON.mesh_util import load_checkpoint
model = load_checkpoint(model, cfg)

pbar = tqdm(dataset)
print(colored(f'Dataset Size: {len(dataset)}', 'green'))

for data in pbar:
    if data['name'] != '10':
        continue
    pbar.set_description(f"{data['name']}")

    in_tensor = {'image': data['image'], 'image_tensor': data['image_tensor'], 'smpl_faces': torch.as_tensor([SMPL.faces]).long(), 'name': data['name']}
    in_tensor['image_tensor_1'] = data['image_tensor_1'].to(device)
    in_tensor['image_tensor_2'] = data['image_tensor_2'].to(device)
    in_tensor['image_tensor_3'] = data['image_tensor_3'].to(device)
    
    in_tensor['image_tensor'] = in_tensor['image_tensor'].to(device)
    in_tensor['image'] = in_tensor['image'].to(device)
    in_tensor['smpl_faces'] = in_tensor['smpl_faces'].to(device)

    ##################################################
    ##### LVD Encoder Run #####
    ##################################################

    with torch.no_grad():
        input_points = torch.zeros(1, 6890, 3).to(device)
        ## im_feat_list 생성
        LVD_encoder(in_tensor['image_tensor'])
        iters = 10
        inds = np.arange(6890)
        for it in range(iters):
            pred_dist = LVD_encoder.query_test(input_points)[None]
            input_points = - pred_dist + input_points

        pred_mesh_0 = input_points[0].cpu().data.numpy()#.view(1, -1, 3)
        
    with torch.no_grad():
        input_points_1 = torch.zeros(1, 6890, 3).to(device)
        ## im_feat_list 생성
        LVD_encoder(in_tensor['image_tensor_1'])
        iters = 10
        inds = np.arange(6890)
        for it in range(iters):
            pred_dist = LVD_encoder.query_test(input_points_1)[None]
            input_points_1 = - pred_dist + input_points_1

        
        
    with torch.no_grad():
        input_points_2 = torch.zeros(1, 6890, 3).to(device)
        ## im_feat_list 생성
        LVD_encoder(in_tensor['image_tensor_2'])
        iters = 10
        inds = np.arange(6890)
        for it in range(iters):
            pred_dist = LVD_encoder.query_test(input_points_2)[None]
            input_points_2 = - pred_dist + input_points_2

        
        
    with torch.no_grad():
        input_points_3 = torch.zeros(1, 6890, 3).to(device)
        ## im_feat_list 생성
        LVD_encoder(in_tensor['image_tensor_3'])
        iters = 10
        inds = np.arange(6890)
        for it in range(iters):
            pred_dist = LVD_encoder.query_test(input_points_3)[None]
            input_points_3 = - pred_dist + input_points_3

        index_1 = torch.tensor([2, 1, 0]).unsqueeze(0).unsqueeze(0)
        index_1 = index_1.expand(input_points.size(0), input_points.size(1), index_1.size(2))
        index_1 = index_1.to(device)
        
        #pred_mesh = pred_mesh / pred_mesh.max()
        #pred_mesh_ = pred_mesh.cpu().data.numpy()
        input_points_1 = input_points_1.gather(2, index_1) * torch.tensor([1.0, 1.0, -1.0]).to(device)
        input_points_2 = input_points_2 * torch.tensor([-1.0, 1.0, -1.0]).to(device)
        input_points_3 = input_points_3.gather(2, index_1) * torch.tensor([-1.0, 1.0, 1.0]).to(device)
        
        pred_mesh_1 = input_points_1[0].cpu().data.numpy()
        pred_mesh_2 = input_points_2[0].cpu().data.numpy()
        pred_mesh_3 = input_points_3[0].cpu().data.numpy()

        mean_pred_mesh = (pred_mesh_0 + pred_mesh_1 + pred_mesh_2 + pred_mesh_3) / 4
        min_pred_mesh = pred_mesh_0 * (abs(pred_mesh_0 - mean_pred_mesh) >= abs(pred_mesh_1 - mean_pred_mesh)) + pred_mesh_1 * (abs(pred_mesh_0 - mean_pred_mesh) < abs(pred_mesh_1 - mean_pred_mesh))
        min_pred_mesh = min_pred_mesh * (abs(min_pred_mesh - mean_pred_mesh) >= abs(pred_mesh_2 - mean_pred_mesh)) + pred_mesh_2 * (abs(min_pred_mesh - mean_pred_mesh) < abs(pred_mesh_2 - mean_pred_mesh))
        min_pred_mesh = min_pred_mesh * (abs(min_pred_mesh - mean_pred_mesh) >= abs(pred_mesh_3 - mean_pred_mesh)) + pred_mesh_3 * (abs(min_pred_mesh - mean_pred_mesh) < abs(pred_mesh_3 - mean_pred_mesh))

        pred_mesh = (pred_mesh_0 * 2 + pred_mesh_1 + pred_mesh_2 * 2 + pred_mesh_3 - min_pred_mesh) / 5
        
        m = trimesh.Trimesh(pred_mesh_1, in_tensor['smpl_faces'].detach().cpu()[0], process = False, maintains_order = True)
        m.export('./result/0_LVD_1_%s.obj'%data['name'])
        m = trimesh.Trimesh(pred_mesh_2, in_tensor['smpl_faces'].detach().cpu()[0], process = False, maintains_order = True)
        m.export('./result/0_LVD_B_%s.obj'%data['name'])
        m = trimesh.Trimesh(pred_mesh_3, in_tensor['smpl_faces'].detach().cpu()[0], process = False, maintains_order = True)
        m.export('./result/0_LVD_3_%s.obj'%data['name'])
        
        m = trimesh.Trimesh(pred_mesh, in_tensor['smpl_faces'].detach().cpu()[0], process = False, maintains_order = True)
        m.export('./result/0_LVD_F_%s.obj'%data['name'])

    ##################################################
    ##### LVD like SMPL result #####
    ##################################################

    if mode_smpl == 'icon':
        
        class OptimizationSMPL(torch.nn.Module):
            def __init__(self):
                super(OptimizationSMPL, self).__init__()
                
                # self.orient = torch.nn.Parameter(torch.zeros(1, 1, 3, 3).to(device)) # 1, 1, 3, 3
                self.theta = torch.nn.Parameter(torch.zeros(1, 72).to(device)) # 1, 23, 3, 3
                self.beta = torch.nn.Parameter((torch.zeros(1, 300).to(device))) # 1, 10
                self.trans = torch.nn.Parameter(torch.zeros(1, 3).to(device)) # 1, 3
                self.scale = torch.nn.Parameter(torch.ones(1).to(device)*90) # 1 * 90

            def forward(self):
                # return self.orient, self.pose, self.beta, self.trans, self.scale
                return self.theta, self.beta, self.trans, self.scale
        
        e3 = torch.eye(3).float().to(device)
        parameters_smpl = OptimizationSMPL().to(device)
        optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=LVD_lr)
        
        #prior = MaxMixturePrior(prior_folder='./model_save/LVD_SMPL/', num_gaussians=8) #.get_gmm_prior()
        #prior = prior.to(device)
        
    elif mode_smpl == 'lvd':
        
        class OptimizationSMPL(torch.nn.Module):
            def __init__(self):
                super(OptimizationSMPL, self).__init__()

                self.pose = torch.nn.Parameter(torch.zeros(1, 72).to(device))
                self.beta = torch.nn.Parameter((torch.zeros(1, 300).to(device)))
                self.trans = torch.nn.Parameter(torch.zeros(1, 3).to(device))
                self.scale = torch.nn.Parameter(torch.ones(1).to(device)*90)

            def forward(self):
                return self.pose, self.beta, self.trans, self.scale
        
        parameters_smpl = OptimizationSMPL().to(device)
        optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=LVD_lr)
        
        prior = MaxMixturePrior(prior_folder='./model_save/LVD_SMPL/', num_gaussians=8) #.get_gmm_prior()
        prior = prior.to(device)
        
        #optimed_pose = torch.nn.Parameter(torch.zeros(1, 72).to(device))
        #optimed_betas = torch.nn.Parameter(torch.zeros(1, 300).to(device))
        #optimed_trans = torch.nn.Parameter(torch.zeros(1, 3).to(device))
        #optimed_scale = torch.nn.Parameter(torch.ones(1).to(device) * 90)
        
        #optimizer_smpl = torch.optim.Adam(
        #    [optimed_pose, optimed_betas, optimed_trans, optimed_scale],
        #    lr=LVD_lr, 
        #)
        
        
    iterations = tqdm(range(LVD_loop))
    pred_mesh_torch = torch.FloatTensor(pred_mesh).to(device)

    factor_beta_reg = 0.2
    
    for i in iterations:
        
        if mode_smpl == 'icon':
            optimed_theta, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
            
            Rs = batch_rodrigues(optimed_theta.view(-1, 3)).view(-1, 24, 3, 3)
            optimed_pose = (Rs[:, 1:, :, :]).sub(1.0, e3)#.view(-1, 207)
            optimed_orient = Rs[:, :1, :, :]
            
            vertices_smpl = (SMPL(body_pose=optimed_pose, betas=optimed_betas, global_orient = optimed_orient, pose2rot = False).vertices + optimed_trans)*optimed_scale
        elif mode_smpl == 'lvd':
            optimed_pose, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
            vertices_smpl = (SMPL.forward(theta=optimed_pose, beta=optimed_betas, get_skin=True)[0][0] + optimed_trans)*optimed_scale
        
        distances = torch.abs(pred_mesh_torch - vertices_smpl)
        loss = distances.mean()
        
        #######################################################################
        
        pre_mesh = vertices_smpl.view(1, -1, 3) / 100# * torch.tensor([1.0, -1.0, -1.0]).to(device)
        #pre_mesh = (vertices_smpl / vertices_smpl.max()).cpu().data.numpy()
        
        # render = Render(size = 512, device = device)
        # render.load_meshes(pre_mesh, in_tensor['smpl_faces'].detach().cpu()[0])
        # in_tensor["T_normal_F"], in_tensor["T_normal_1"], in_tensor["T_normal_B"], in_tensor["T_normal_3"] = render.get_rgb_image(cam_ids=[0, 1, 2, 3])
        # T_mask_F, T_mask_1, T_mask_B, T_mask_3 = render.get_silhouette_image(cam_ids=[0, 1, 2, 3])
        

        
        render = Render(size = 512, device = device)
        render.load_meshes(pre_mesh, in_tensor['smpl_faces'].detach().cpu()[0])
        in_tensor["T_normal_F"], _ = render.get_rgb_image()
        T_mask_F, _ = render.get_silhouette_image()
        
        render.load_meshes(pre_mesh * torch.tensor([-1.0, 1.0, -1.0]).to(device), in_tensor['smpl_faces'].detach().cpu()[0])
        in_tensor["T_normal_B"], _ = render.get_rgb_image()
        T_mask_B, _ = render.get_silhouette_image()
        

        
        render.load_meshes(pre_mesh.gather(2, index_1) * torch.tensor([-1.0, 1.0, 1.0]).to(device), in_tensor['smpl_faces'].detach().cpu()[0])
        in_tensor["T_normal_1"], _ = render.get_rgb_image()
        T_mask_1, _ = render.get_silhouette_image()
        
        render.load_meshes(pre_mesh.gather(2, index_1) * torch.tensor([1.0, 1.0, -1.0]).to(device), in_tensor['smpl_faces'].detach().cpu()[0])
        in_tensor["T_normal_3"], _ = render.get_rgb_image()
        T_mask_3, _ = render.get_silhouette_image()
        


        data_F = {
            'image': data['image'].to(device), 
            'T_normal_F': in_tensor['T_normal_F'].to(device), 
            'T_normal_B': in_tensor['T_normal_B'].to(device)
        }
        data_B = {
            'image': data['image_2'].to(device), 
            'T_normal_F': in_tensor['T_normal_B'].to(device), 
            'T_normal_B': in_tensor['T_normal_F'].to(device)
        }
        data_1 = {
            'image': data['image_1'].to(device), 
            'T_normal_F': in_tensor['T_normal_1'].to(device), 
            'T_normal_B': in_tensor['T_normal_3'].to(device)
        }
        data_3 = {
            'image': data['image_3'].to(device), 
            'T_normal_F': in_tensor['T_normal_3'].to(device), 
            'T_normal_B': in_tensor['T_normal_1'].to(device)
        }
        
        with torch.no_grad():
            in_tensor['normal_F'], _ = model.netG.normal_filter(data_F)
            in_tensor['normal_1'], _ = model.netG.normal_filter(data_1)
            in_tensor['normal_B'], _ = model.netG.normal_filter(data_B)
            in_tensor['normal_3'], _ = model.netG.normal_filter(data_3)
        
        diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
        diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])
        diff_1_smpl = torch.abs(in_tensor["T_normal_1"] - in_tensor["normal_1"])
        diff_3_smpl = torch.abs(in_tensor["T_normal_3"] - in_tensor["normal_3"])
        
        # normal loss
        normal_loss = (diff_F_smpl + diff_B_smpl + diff_1_smpl + diff_3_smpl).mean()
        
        # silhouette loss
        smpl_arr = torch.cat([T_mask_F, T_mask_B, T_mask_1, T_mask_3], dim=-1)[0]
        gt_arr = torch.cat([in_tensor["normal_F"][0], in_tensor["normal_B"][0], in_tensor["normal_1"][0], in_tensor["normal_3"][0]], dim=2).permute(1, 2, 0)
        gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
        bg_color = (torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device))
        gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
        diff_S = torch.abs(smpl_arr - gt_arr)
        silhouette_loss = diff_S.mean()

        #######################################################################
        
        if mode_smpl == 'icon':
            prior_loss = 0#prior.forward(optimed_pose[:, 3:], optimed_betas)
        elif mode_smpl == 'lvd':
            prior_loss = prior.forward(optimed_pose[:, 3:], optimed_betas)
        beta_loss = (optimed_betas**2).mean()
        loss = loss + prior_loss * 0.01 + beta_loss * factor_beta_reg + normal_loss + silhouette_loss
        
        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()
        #scheduler_smpl.step(loss)
        
        for param_group in optimizer_smpl.param_groups:
            param_group['lr'] = LVD_lr*(LVD_loop-i)/LVD_loop
            
        pbar_desc = "SMPL fitting LVD --- "
        pbar_desc += f"Total loss: {loss.item():.5f}"
        iterations.set_description(pbar_desc)
        
        
    with torch.no_grad():
        if mode_smpl == 'icon':
            optimed_theta, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
            
            Rs = batch_rodrigues(optimed_theta.view(-1, 3)).view(-1, 24, 3, 3)
            optimed_pose = (Rs[:, 1:, :, :]).sub(1.0, e3)#.view(-1, 207)
            optimed_orient = Rs[:, :1, :, :]
            
            vertices_smpl = (SMPL(body_pose=optimed_pose, betas=optimed_betas, global_orient = optimed_orient, pose2rot = False).vertices + optimed_trans)*optimed_scale
            pred_mesh = vertices_smpl[0].cpu().data.numpy()
            
        elif mode_smpl == 'lvd':
            optimed_pose, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
            vertices_smpl = (SMPL.forward(theta=optimed_pose, beta=optimed_betas, get_skin=True)[0][0] + optimed_trans)*optimed_scale
            pred_mesh = vertices_smpl.cpu().data.numpy()

    m = trimesh.Trimesh(pred_mesh, in_tensor['smpl_faces'].detach().cpu()[0], process = False, maintains_order = True)
    m.export('./result/1_LVD_%s.obj'%data['name'])
    
    name = data['name']
    if name == '0':
        jf_text1 = '184'
    if name == '1':
        jf_text1 = '169'
    if name == '2':
        jf_text1 = '173'
    if name == '3':
        jf_text1 = '175'
    if name == '4':
        jf_text1 = '165'
    if name == '5':
        jf_text1 = '168'
    if name == '6':
        jf_text1 = '172'
    if name == '7':
        jf_text1 = '158'
    if name == '8':
        jf_text1 = '167'
    if name == '9':
        jf_text1 = '175'
    if name == '10':
        jf_text1 = '168'
    
    lvd_person = pywavefront.Wavefront(
        './result/1_LVD_%s.obj'%name, 
        create_materials=True,
        collect_faces=True
    )
    
    lvd_faces = np.array(lvd_person.mesh_list[0].faces)
    lvd_vertices = np.array(lvd_person.vertices) / 100
    
    lvd_body = Body3D(lvd_vertices, lvd_faces)
    
    lvd_body_measurements = lvd_body.getMeasurements()
    
    lvd_height = lvd_body.height()
    lvd_weight = lvd_body.weight()
    _, _, lvd_neck_length = lvd_body.neck()
    _, _, lvd_bicep_length = lvd_body.bicep()
    _, _, lvd_chest_length = lvd_body.chest()
    _, _, lvd_waist_length = lvd_body.waist()
    _, _, lvd_hip_length = lvd_body.hip()
    _, _, lvd_thigh_length = lvd_body.thighOutline()
    
    lvd_ratio = copy(float(jf_text1) / lvd_height)
    lvd_height = lvd_height * lvd_ratio
    lvd_weight = lvd_weight * lvd_ratio * lvd_ratio / 10000
    lvd_neck_length = lvd_neck_length * lvd_ratio
    lvd_bicep_length = lvd_bicep_length * lvd_ratio
    lvd_chest_length = lvd_chest_length * lvd_ratio
    lvd_waist_length = lvd_waist_length * lvd_ratio
    lvd_hip_length = lvd_hip_length * lvd_ratio
    lvd_thigh_length = lvd_thigh_length * lvd_ratio
    lvd_BMI = lvd_weight / (lvd_height / 100) ** 2
    
    if lvd_BMI <= 18.5:
        BMI_result = '저체중'
    elif lvd_BMI < 23.0:
        BMI_result = '정상'
    elif lvd_BMI < 25.0:
        BMI_result = '과체중'
    else:
        BMI_result  = '비만'
        
    lvd_height = round(lvd_height, 2)
    lvd_weight = round(lvd_weight, 2)
    lvd_neck_length = round(lvd_neck_length, 2)
    lvd_bicep_length = round(lvd_bicep_length, 2)
    lvd_chest_length = round(lvd_chest_length, 2)
    lvd_waist_length = round(lvd_waist_length, 2)
    lvd_hip_length = round(lvd_hip_length, 2)
    lvd_thigh_length = round(lvd_thigh_length, 2)
    lvd_BMI = round(lvd_BMI, 2)

    
    output = {
        "text": [
            {
                "키": lvd_height,
                "몸무게": lvd_weight,
                "목 둘레": lvd_neck_length,
                "팔뚝 둘레": lvd_bicep_length,
                "가슴 둘레": lvd_chest_length,
                "배 둘레": lvd_waist_length,
                "엉덩이 둘레": lvd_hip_length,
                "허벅지 둘레": lvd_thigh_length,
                "비만지수" : lvd_BMI,
                "비만 결과" : BMI_result,
            }
        ]
    }
    
    print()
    print(name)
    print(output)
    print()
    print()
    

    
    
    ##################################################
    ##### SMPL in ICON #####
    ##################################################
    
    ##### 일단 이 부분 생략 후 결과 비교 
    
    ##################################################
    ##### Clothed mesh (After SMPL in ICON) #####
    ##################################################
    
    
    #in_tensor = {"smpl_faces": data["smpl_faces"], "image": data["image"]}
    #in_tensor["smpl_joint"] = smpl_joints[:, :24, :]
    
    ##### F, B 생성
    
    #in_tensor
    #dict_keys([
        # 'smpl_faces', 
        # 'image', 
        # 'smpl_joint'])  -> key point
        
    
    in_tensor['smpl_verts'] = vertices_smpl.view(1, -1, 3) / 100# * torch.tensor([1.0, -1.0, -1.0]).to(device)
    
    #in_tensor['smpl_verts'] = ((vertices_smpl / vertices_smpl.max()) * torch.tensor([1.0, 1.0, -1.0]).to(device)).unsqueeze(0) * torch.tensor([1.0, -1.0, -1.0]).to(device)
    
    #pre_mesh = (vertices_smpl / vertices_smpl.max()).cpu().data.numpy()
    
    ##################################################
    ##### 3d to 2d rendering image #####
    ##################################################
    
    
    render.load_meshes(pre_mesh, in_tensor['smpl_faces'].detach().cpu()[0])
    t_3d, _ = render.get_rgb_image()
    render.load_meshes(pre_mesh * torch.tensor([-1.0, 1.0, -1.0]).to(device), in_tensor['smpl_faces'].detach().cpu()[0])
    f_3d, _ = render.get_rgb_image()
    render.load_meshes(pre_mesh.gather(2, index_1) * torch.tensor([-1.0, 1.0, 1.0]).to(device), in_tensor['smpl_faces'].detach().cpu()[0])
    t_3d_1, _ = render.get_rgb_image()
    render.load_meshes(pre_mesh.gather(2, index_1) * torch.tensor([1.0, 1.0, -1.0]).to(device), in_tensor['smpl_faces'].detach().cpu()[0])
    t_3d_3, _ = render.get_rgb_image()
    
    tt = ((t_3d[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    ff = ((f_3d[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    t1 = ((t_3d_1[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    t3 = ((t_3d_3[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    Image.fromarray(tt).save('./result_img/%s_smpl_forward_3d.png'%data['name'])
    Image.fromarray(ff).save('./result_img/%s_smpl_back_3d.png'%data['name'])
    Image.fromarray(t1).save('./result_img/%s_smpl_f_1_3d.png'%data['name'])
    Image.fromarray(t3).save('./result_img/%s_smpl_f_3_3d.png'%data['name'])
    
    
    
    ##################################################
    ##### 3d to generative 2d image #####
    ##################################################
    
    with torch.no_grad():
        #t_gan, f_gan = normal_net(data)
        t_gan, _ = model.netG.normal_filter(in_tensor)
        f_gan, _ = model.netG.normal_filter(data_B)
        t_gan_1, _ = model.netG.normal_filter(data_1)
        t_gan_3, _ = model.netG.normal_filter(data_3)
    
    tt_gan = ((t_gan[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    ff_gan = ((f_gan[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    t1_gan = ((t_gan_1[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    t3_gan = ((t_gan_3[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    Image.fromarray(tt_gan).save('./result_img/%s_img_forward_gan.png'%data['name'])
    Image.fromarray(ff_gan).save('./result_img/%s_img_back_gan.png'%data['name'])
    Image.fromarray(t1_gan).save('./result_img/%s_img_f_1_gan.png'%data['name'])
    Image.fromarray(t3_gan).save('./result_img/%s_img_f_3_gan.png'%data['name'])
    
    
    
    
    ##################################################
    ##### ICON 뒷 부분 #####
    ##################################################
    ##################################################
    ##### ICON 뒷 부분 #####
    ##################################################
    ##################################################
    ##### ICON 뒷 부분 #####
    ##################################################
    
    in_tensor["smpl_verts"] = (vertices_smpl / 100 * torch.tensor([1.0, -1.0, -1.0]).to(device)).unsqueeze(0) * torch.tensor([1.0, 1.0, -1.0]).to(device)
    
    in_tensor.update(dataset.compute_vis_cmap(in_tensor["smpl_verts"][0], in_tensor["smpl_faces"][0], device))

    in_tensor.update({
        "smpl_norm": compute_normal_batch(in_tensor["smpl_verts"], in_tensor["smpl_faces"])
    })


    with torch.no_grad():
        verts_pr, faces_pr, _ = model.test_single(in_tensor)

    recon_obj = trimesh.Trimesh(verts_pr,
                                faces_pr,
                                process=False,
                                maintains_order=True)
    
    recon_obj.export(os.path.join('./result/2_ICON_%s_recon.obj'%data['name']))


    verts_refine, faces_refine = remesh(os.path.join('./result/2_ICON_%s_recon.obj'%data['name']), 0.5, device)
    
    ##################################################
    ##### ICON model 실행 부분 #####
    ##################################################
    ##################################################
    ##### ICON model 실행 부분 #####
    ##################################################
    ##################################################
    ##### ICON model 실행 부분 #####
    ##################################################
    
    
    
    
    
    
    from models_ICON_IMGGAN.local_affine import LocalAffine
    from models_SMPL_ICON.mesh_util import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing

    mesh_pr = Meshes(verts_refine, faces_refine).to(device)
    local_affine_model = LocalAffine(mesh_pr.verts_padded().shape[1], mesh_pr.verts_padded().shape[0], mesh_pr.edges_packed()).to(device)
    optimizer_cloth = torch.optim.Adam(
        [{
            'params': local_affine_model.parameters()
        }],
        lr=1e-4,
        amsgrad=True)

    scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cloth,
        mode="min",
        factor=0.1,
        verbose=0,
        min_lr=1e-5,
        patience=5,
    )
    
    
    with torch.no_grad():
        rotate_recon_lst = render.get_rgb_image(cam_ids=[0, 1, 2, 3])

    final = None
    
    loop_cloth = tqdm(range(loop_cloth__))
    
    for i in loop_cloth:
    
        optimizer_cloth.zero_grad()
        
        deformed_verts, stiffness, rigid = local_affine_model(verts_refine.to(device), return_stiff=True)
        mesh_pr = mesh_pr.update_padded(deformed_verts)
        
        edge_loss = mesh_edge_loss(mesh_pr)
        nc_loss = mesh_normal_consistency(mesh_pr)
        laplacian_loss = mesh_laplacian_smoothing(mesh_pr, method = 'uniform')
        
        index_2 = torch.tensor([2, 1, 0]).unsqueeze(0).unsqueeze(0)
        index_2 = index_2.expand(mesh_pr.verts_padded().size(0), mesh_pr.verts_padded().size(1), index_2.size(2))
        index_2 = index_2.to(device)
        
        render.load_meshes(mesh_pr.verts_padded(), mesh_pr.faces_padded())
        in_tensor["P_normal_F"], _ = render.get_rgb_image()
        
        render.load_meshes(mesh_pr.verts_padded() * torch.tensor([-1.0, 1.0, -1.0]).to(device), mesh_pr.faces_padded())
        in_tensor["P_normal_B"], _ = render.get_rgb_image()
        
        render.load_meshes(mesh_pr.verts_padded().gather(2, index_2) * torch.tensor([-1.0, 1.0, 1.0]).to(device), mesh_pr.faces_padded())
        in_tensor["P_normal_1"], _ = render.get_rgb_image()
        
        render.load_meshes(mesh_pr.verts_padded().gather(2, index_2) * torch.tensor([1.0, 1.0, -1.0]).to(device), mesh_pr.faces_padded())
        in_tensor["P_normal_3"], _ = render.get_rgb_image()
        
        
        
        diff_F_cloth = torch.abs(in_tensor["P_normal_F"] - in_tensor["normal_F"])
        diff_B_cloth = torch.abs(in_tensor["P_normal_B"] - in_tensor["normal_B"])
        diff_1_cloth = torch.abs(in_tensor["P_normal_1"] - in_tensor["normal_1"])
        diff_3_cloth = torch.abs(in_tensor["P_normal_3"] - in_tensor["normal_3"])
        
        cloth_loss = (diff_F_cloth + diff_F_cloth + diff_B_cloth + diff_B_cloth + diff_1_cloth + diff_3_cloth).mean()
        stiffness_loss = torch.mean(stiffness)
        rigid_loss = torch.mean(rigid)
        
        loss = torch.tensor(0.0, requires_grad=True).to(device)
        
        loss = loss + 1e1 * cloth_loss + 1e5 * stiffness_loss + 1e5 * rigid_loss + 1e2 * laplacian_loss
        
        loss.backward(retain_graph=True)
        optimizer_cloth.step()
        scheduler_cloth.step(loss)
        
        pbar_desc = "Human fitting ICON --- "
        pbar_desc += f"Total loss: {loss.item():.5f}"
        loop_cloth.set_description(pbar_desc)
        
        
    # final = trimesh.Trimesh(
    #     mesh_pr.verts_packed().detach().squeeze(0).cpu(),
    #     mesh_pr.faces_packed().detach().squeeze(0).cpu(),
    #     process=False,
    #     maintains_order=True)
    # final_colors = query_color_4v(
    #     mesh_pr.verts_packed().detach().squeeze(0).cpu(),
    #     mesh_pr.faces_packed().detach().squeeze(0).cpu(),
    #     data["image_0"].to(device),
    #     data["image_2"].to(device),
    #     device=device,
    # )
    final = trimesh.Trimesh(
        mesh_pr.verts_packed().detach().squeeze(0).cpu(),
        mesh_pr.faces_packed().detach().squeeze(0).cpu(),
        process=False,
        maintains_order=True)
    final_colors = query_color(
        mesh_pr.verts_packed().detach().squeeze(0).cpu(),
        mesh_pr.faces_packed().detach().squeeze(0).cpu(),
        in_tensor["image"],
        device=device,
    )
    
    final.visual.vertex_colors = final_colors
    final.export(os.path.join('./result/4_ICON_%s_refine.obj'%data['name']))
    
    index_3 = torch.tensor([2, 1, 0]).unsqueeze(0).unsqueeze(0)
    index_3 = index_3.expand(mesh_pr.verts_padded().size(0), mesh_pr.verts_padded().size(1), index_3.size(2))
    index_3 = index_3.to(device)
    
    render.load_meshes(mesh_pr.verts_packed().unsqueeze(0), mesh_pr.faces_packed().detach().cpu())
    t_3d, _ = render.get_rgb_image()
    render.load_meshes(mesh_pr.verts_packed().unsqueeze(0) * torch.tensor([-1.0, 1.0, -1.0]).to(device), mesh_pr.faces_packed().detach().cpu())
    f_3d, _ = render.get_rgb_image()
    render.load_meshes(mesh_pr.verts_packed().unsqueeze(0).gather(2, index_3) * torch.tensor([-1.0, 1.0, 1.0]).to(device), mesh_pr.faces_packed().detach().cpu())
    t_3d_1, _ = render.get_rgb_image()
    render.load_meshes(mesh_pr.verts_packed().unsqueeze(0).gather(2, index_3) * torch.tensor([1.0, 1.0, -1.0]).to(device), mesh_pr.faces_packed().detach().cpu())
    t_3d_3, _ = render.get_rgb_image()
    
    tt = ((t_3d[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    ff = ((f_3d[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    t1 = ((t_3d_1[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    t3 = ((t_3d_3[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    Image.fromarray(tt).save('./result_img/%s_final_forward_3d.png'%data['name'])
    Image.fromarray(ff).save('./result_img/%s_final_back_3d.png'%data['name'])
    Image.fromarray(t1).save('./result_img/%s_final_f_1_3d.png'%data['name'])
    Image.fromarray(t3).save('./result_img/%s_final_f_3_3d.png'%data['name'])
    
