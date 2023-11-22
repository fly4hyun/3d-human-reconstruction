
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
        self.image_dir = './woman/img_data'
        self.obj_dir = './woman/obj_data_smpl'
        
        keep_list = sorted(glob.glob(f'{self.image_dir}/*'))
        img_format = ['jpg', 'png', 'jpeg', "JPG", 'bmp', 'exr']
        keep_list = [item for item in keep_list if item.split('.')[-1] in img_format]
        
        self.image_list = sorted([item[:-6] for item in keep_list if item.split(".")[-1] in img_format])
        self.image_list = list(set(self.image_list))
        
        self.pose_mp = mp.solutions.pose.Pose(
            static_image_mode=True, 
            model_complexity=2, 
            enable_segmentation=True, 
            min_detection_confidence=0.5
            )

        self.smpl_data = SMPLX()
        
        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
        self.hps.eval()



    def __len__(self):
        return len(self.image_list)
    
    
    
    
    def compute_vis_cmap(self, smpl_verts, smpl_faces, device):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, -z, torch.as_tensor(smpl_faces).long())
        smpl_cmap = self.smpl_data.cmap_smpl_vids('smpl')

        return {
            'smpl_vis': smpl_vis.unsqueeze(0).to(device),
            'smpl_cmap': smpl_cmap.unsqueeze(0).to(device),
            'smpl_verts': smpl_verts.unsqueeze(0)
        }
    
    
    
    
    
    def __getitem__(self, index):
        
        img_path = self.image_list[index]
        img_name = img_path.split('/')[-1].rsplit('.', 1)[0]
        obj_path = self.obj_dir + '/' + img_name + '.obj'
        
        img_0, img_hps_0, img_ori, img_mask, uncrop_param = process_image(img_path + '_0.png', 'pymaf', 512)
        img_1, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_1.png', 'pymaf', 512)
        img_2, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_2.png', 'pymaf', 512)
        img_3, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_3.png', 'pymaf', 512)
        
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
        
        mesh_vert = 1.8 * ((mesh_vert - mesh_vert[0, :, 1].min()) / (mesh_vert[0, :, 1].max() - mesh_vert[0, :, 1].min())) - 0.89

        mesh_vert[0, :, 0] = mesh_vert[0, :, 0] - mesh_vert[0, :, 0].mean()
        mesh_vert[0, :, 2] = mesh_vert[0, :, 2] - mesh_vert[0, :, 2].mean()
        
        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            preds_dict = self.hps.forward(img_hps_0.to(self.device))
        
        output = preds_dict['smpl_out'][-1]
        scale, tranX, tranY = output['theta'][0, :3]
        
        data_dict = {
            'name' : img_name, 
            'image_0' : img_0.unsqueeze(0), 
            'image_1' : img_1.unsqueeze(0), 
            'image_2' : img_2.unsqueeze(0), 
            'image_3' : img_3.unsqueeze(0), 
            'face' : mesh_face, 
            'verts' : mesh_vert,
            'orient' : output['rotmat'][:, 0:1], 
            'pose' : output['rotmat'][:, 1:], 
            'beta' : output['pred_shape'], 
            'trans' : torch.tensor([tranX, tranY, 0.0]).unsqueeze(0).float(), 
            'scale' : scale, 
        }

        return data_dict







