




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


from b0_LVD_data_loader import TestDataset
from models_LVD_encoder.LVD_encoder import Network
import models_SMPL_ICON.smplx as smplx
from models_SMPL_ICON.mesh_util import SMPLX, remesh
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

##################################################
##### parameter setting #####
##################################################

mode_smpl = 'lvd'
# icon
# lvd

##### SMPL #####

smpl_type = 'smpl'
smpl_gender = 'neutral'
# male
# female
# neutral

##### LVD Encoder #####

LVD_encoder_epoch_label = 1000
LVD_encoder_network_label = 'img_encoder'
LVD_encoder_save_dir = './model_save/LVD_encoder'
LVD_encoder_load_filename = 'net_epoch_%s_id_%s.pth' % (LVD_encoder_epoch_label, LVD_encoder_network_label)
LVD_encoder_load_path = os.path.join(LVD_encoder_save_dir, LVD_encoder_load_filename)

if mode_smpl == 'icon':
    LVD_loop = 10000
    LVD_lr = 1e-2
elif mode_smpl == 'lvd':
    LVD_loop = 40
    LVD_lr = 1e-1

loop_cloth__ = 500



##### device setting #####

device = torch.device('cuda:0')

##################################################
##### image import #####
##################################################

render = Render(size = 512, device = device)
dataset = TestDataset()







##################################################
##### model import #####
##################################################

##### SMPL import #####

##### SMPL import (ICON) #####

if mode_smpl == 'icon':
    smpl_data = SMPLX()
    get_smpl_model = lambda smpl_type, smpl_gender: smplx.create(
        model_path = smpl_data.model_dir,
        gender = smpl_gender,
        model_type = smpl_type,
        ext = 'npz'
        )

    SMPL = get_smpl_model(smpl_type, smpl_gender).to(device)
    faces = SMPL.faces

##### SMPL import (LVD) #####

elif mode_smpl == 'lvd':
    SMPL = SMPL('./model_save/LVD_SMPL/neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).to(device)

##### LVD Encoder import #####

LVD_encoder = Network(input_channels = 4, pred_dimensions = 6890 * 3)
LVD_encoder.load_state_dict(torch.load(LVD_encoder_load_path))
LVD_encoder = LVD_encoder.to(device)

##### ICON image GAN import #####

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

        # mean_pred_mesh = (pred_mesh_0 + pred_mesh_1 + pred_mesh_2 + pred_mesh_3) / 4
        # min_pred_mesh = pred_mesh_0 * (abs(pred_mesh_0 - mean_pred_mesh) >= abs(pred_mesh_1 - mean_pred_mesh)) + pred_mesh_1 * (abs(pred_mesh_0 - mean_pred_mesh) < abs(pred_mesh_1 - mean_pred_mesh))
        # min_pred_mesh = min_pred_mesh * (abs(min_pred_mesh - mean_pred_mesh) >= abs(pred_mesh_2 - mean_pred_mesh)) + pred_mesh_2 * (abs(min_pred_mesh - mean_pred_mesh) < abs(pred_mesh_2 - mean_pred_mesh))
        # min_pred_mesh = min_pred_mesh * (abs(min_pred_mesh - mean_pred_mesh) >= abs(pred_mesh_3 - mean_pred_mesh)) + pred_mesh_3 * (abs(min_pred_mesh - mean_pred_mesh) < abs(pred_mesh_3 - mean_pred_mesh))

        # pred_mesh = (pred_mesh_0 * 2 + pred_mesh_1 + pred_mesh_2 * 2 + pred_mesh_3 - min_pred_mesh) / 5
        pred_mesh = (pred_mesh_0 + pred_mesh_1 + pred_mesh_2 + pred_mesh_3) / 4
        
        pred_mesh__ = pred_mesh

        m = trimesh.Trimesh(pred_mesh__, in_tensor['smpl_faces'].detach().cpu()[0], process = False, maintains_order = True)
        m.export('./woman/obj_smpl/test_%s.obj'%data['name'])

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
    m.export('./woman/obj_smpl/%s.obj'%data['name'])
    
    asdfasdf