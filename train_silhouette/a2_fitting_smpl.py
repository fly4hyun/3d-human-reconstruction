




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


from a2_train_data_loader import TestDataset
from models_SMPL_LVD.SMPL import SMPL

from models_SMPL_LVD.prior import SMPLifyAnglePrior, MaxMixturePrior












##################################################
##### parameter setting #####
##################################################

# def loss_smpl(output, label):
#     output_out = output.unsqueeze(2)
#     label_out = label.unsqueeze(1)
#     loss_out = ((output_out-label_out) ** 2).sum(-1)
#     loss_out = loss_out.min(-1)[0]
    
#     output_la = output.unsqueeze(1)
#     label_la = label.unsqueeze(2)
#     loss_la = ((output_la-label_la) ** 2).sum(-1)
#     loss_la = loss_la.min(-1)[0]

#     return 3 * loss_out.sum() + loss_la.sum()







class OptimizationSMPL(torch.nn.Module):
    def __init__(self, device):
        super(OptimizationSMPL, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 72).to(device))
        self.beta = torch.nn.Parameter((torch.zeros(1, 300).to(device)))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).to(device))
        self.scale = torch.nn.Parameter(torch.ones(1).to(device) * 90)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale





device = torch.device('cuda:0')
dataset = TestDataset(device)



##################################################
##### model import #####
##################################################


SMPL = SMPL('./model_save/LVD_SMPL/neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).to(device)










pbar = tqdm(dataset)

for data in pbar:

    parameters_smpl = OptimizationSMPL(device)
    optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=0.1)
    
    prior = MaxMixturePrior(prior_folder='./model_save/LVD_SMPL/', num_gaussians=8) #.get_gmm_prior()
    prior = prior.to(device)


    print('')
    print(data['name'])
    print('')
    
    
    m = trimesh.load('./woman/obj_smpl/%s.obj'%data['name'])
    
    obj_face = torch.tensor(SMPL.faces).unsqueeze(0)


    iterations = tqdm(range(200))
    pred_mesh_torch = data['verts'].to(device)
    

    factor_beta_reg = 0.2
    aaa = [-1.0, 1.0, 1.0]
    for i_num in iterations:
        

        optimed_pose, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
        vertices_smpl = (SMPL.forward(theta=optimed_pose, beta=optimed_betas, get_skin=True)[0][0] + optimed_trans)*optimed_scale
        vertices_smpl = vertices_smpl.view(1, -1, 3)
        # vertices_smpl = (vertices_smpl - vertices_smpl[0, :, 1].min()) / (vertices_smpl[0, :, 1].max() - vertices_smpl[0, :, 1].min())
        vertices_smpl = 1.8 * (vertices_smpl - vertices_smpl[0, :, 1].min()) / (vertices_smpl[0, :, 1].max() - vertices_smpl[0, :, 1].min()) - 0.89


        distances = torch.abs(pred_mesh_torch - vertices_smpl)
        loss = distances.mean()
        
        pre_mesh = vertices_smpl.view(1, -1, 3) * torch.tensor(aaa).to(device)
        
        
        prior_loss = prior.forward(optimed_pose[:, 3:], optimed_betas)
        beta_loss = (optimed_betas**2).mean()
        loss = loss + prior_loss * 0.01 + beta_loss * factor_beta_reg
        
        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()

        
        pbar_desc = "SMPL fitting --- "
        pbar_desc += f"Total loss: {loss.item():.5f}"
        iterations.set_description(pbar_desc)

        

    with torch.no_grad():
        optimed_pose, optimed_betas, optimed_trans, optimed_scale = parameters_smpl.forward()
        vertices_smpl = (SMPL.forward(theta=optimed_pose, beta=optimed_betas, get_skin=True)[0][0] + optimed_trans)*optimed_scale
        
        vertices_smpl = vertices_smpl.view(1, -1, 3) * torch.tensor(aaa).to(device)
        vertices_smpl = 1.8 * (vertices_smpl - vertices_smpl[0, :, 1].min()) / (vertices_smpl[0, :, 1].max() - vertices_smpl[0, :, 1].min()) - 0.89
        
        print()
        print(vertices_smpl.max(1)[0])
        print(vertices_smpl.min(1)[0])
        
        pred_mesh = vertices_smpl.cpu().data.numpy()
        
        
        
        m = trimesh.Trimesh(pred_mesh[0], obj_face.detach().cpu()[0], process = False, maintains_order = True)
        m.export('./test/test_2.obj')
        # m.export('./test/%s_test_2.obj'%data['name'])
        
    m = trimesh.load('./woman/obj_data/%s.obj'%data['name'])
    m.export('./test/test_0.obj')
    m = trimesh.load('./woman/obj_smpl/%s.obj'%data['name'])
    m.export('./test/test_1.obj')
    # m = trimesh.load('./woman/obj_data/%s.obj'%data['name'])
    # m.export('./test/%s_test_0.obj'%data['name'])
    # m = trimesh.load('./woman/obj_smpl/%s.obj'%data['name'])
    # m.export('./test/%s_test_1.obj'%data['name'])
    
    mesh_vert = torch.FloatTensor([m.vertices])
    print()
    print(mesh_vert.max(1)[0])
    print(mesh_vert.min(1)[0])
    
    asdfasfdasfdasdf