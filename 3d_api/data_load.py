

import glob
import mediapipe as mp
import cv2
import torch
import numpy as np

from models_SMPL_ICON.imutils import process_image
from models_SMPL_ICON.mesh_util import get_visibility, SMPLX

class TestDataset():
    def __init__(self):
        
        self.image_dir = './examples'
        
        keep_list = sorted(glob.glob(f'{self.image_dir}/*'))
        img_format = ['jpg', 'png', 'jpeg', "JPG", 'bmp', 'exr']
        keep_list = [item for item in keep_list if item.split('.')[-1] in img_format]
        
        self.image_list = sorted([item for item in keep_list if item.split(".")[-1] in img_format])
        
        
        self.pose_mp = mp.solutions.pose.Pose(
            static_image_mode=True, 
            model_complexity=2, 
            enable_segmentation=True, 
            min_detection_confidence=0.5
            )

        self.smpl_data = SMPLX()



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
        
        img = cv2.imread(img_path)[:, :, ::-1]
        results_mediapipe = self.pose_mp.process(img)
        mask = (results_mediapipe.segmentation_mask > 0.5) * 255

        img, mask = self.crop_image(img, mask)
        
        img = img / 255
        mask = mask / 255
        img = img - [0.485, 0.456, 0.406]
        img = img / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)

        img[:, mask[:, :] == 0] = 0
        
        imgtensor = torch.FloatTensor(np.concatenate((img, mask[None]))).unsqueeze(0)
        
        #########
        
        img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path, 'pymaf', 512)

        data_dict = {
            'name' : img_name, 
            'image_tensor' : imgtensor, 
            'image' : img_icon.unsqueeze(0)
        }

        return data_dict

    def crop_image(self, img, mask, iamge_size = 256):
        (x, y, w, h) = cv2.boundingRect(np.uint8(mask))
        mask = mask[y:y + h, x:x + w]
        img = img[y:y + h, x:x + w]

        # Prepare new image, with correct size:
        margin = 1.1
        im_size = iamge_size
        clean_im_size = im_size / margin
        size = int((max(w, h) * margin) // 2)
        new_x = size - w // 2
        new_y = size - h // 2
        new_img = np.zeros((size * 2, size * 2, 3))
        new_mask = np.zeros((size * 2, size * 2))
        new_img[new_y:new_y + h, new_x:new_x + w] = img
        new_mask[new_y:new_y + h, new_x:new_x + w] = mask

        # Resizing cropped and centered image to desired size:
        img = cv2.resize(new_img, (im_size,im_size))
        mask = cv2.resize(new_mask, (im_size,im_size))

        return img, mask








