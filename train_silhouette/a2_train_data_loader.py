
import numpy as np
import glob
import mediapipe as mp
import cv2
import torch
import trimesh


from models_SMPL_ICON.imutils import process_image
from models_SMPL_ICON.mesh_util import get_visibility, SMPLX
from models_pymaf.pymaf_net import pymaf_net
from models_pymaf.core import path_config

class TestDataset():
    def __init__(self, device):
        
        self.device = device
        self.obj_dir = './woman/obj_smpl'
        
        keep_list = sorted(glob.glob(f'{self.obj_dir}/*'))
        obj_format = ['obj']
        keep_list = [item for item in keep_list if item.split('.')[-1] in obj_format]
        
        self.obj_list = sorted([item for item in keep_list if item.split(".")[-1] in obj_format])
        self.obj_list = list(set(self.obj_list))




    def __len__(self):
        return len(self.image_list)
    
    
    
    
    
    def __getitem__(self, index):
        
        obj_path = self.obj_list[index]
        obj_name = obj_path.split('/')[-1].rsplit('.', 1)[0]
        
        m = trimesh.load(obj_path)
        
        mesh_face = torch.tensor(m.faces)
        
        #new_verts = []
        #for vert in m.vertices:
        #    if new_verts == []:
        #        new_verts.append(vert)
        #        continue
        #    if (new_verts == vert).any():
        #        continue
        #    new_verts.append(list(vert))
        
        #mesh_vert = torch.FloatTensor([new_verts])
        mesh_vert = torch.FloatTensor([m.vertices])
        
        
        data_dict = {
            'name' : obj_name, 
            'face' : mesh_face, 
            'verts' : mesh_vert,
        }

        return data_dict







