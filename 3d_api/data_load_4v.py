

import glob
import mediapipe as mp
import cv2
import torch
import numpy as np

from models_SMPL_ICON.imutils import process_image
from models_SMPL_ICON.mesh_util import get_visibility, SMPLX

class TestDataset():
    def __init__(self):
        
        self.image_dir = './examples_4v'
        
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
        
        img_0 = cv2.imread(img_path + '_0.png')[:, :, ::-1]
        img_1 = cv2.imread(img_path + '_1.png')[:, :, ::-1]
        img_2 = cv2.imread(img_path + '_2.png')[:, :, ::-1]
        img_3 = cv2.imread(img_path + '_3.png')[:, :, ::-1]
        results_mediapipe_0 = self.pose_mp.process(img_0)
        results_mediapipe_1 = self.pose_mp.process(img_1)
        results_mediapipe_2 = self.pose_mp.process(img_2)
        results_mediapipe_3 = self.pose_mp.process(img_3)
        mask_0 = (results_mediapipe_0.segmentation_mask > 0.5) * 255
        mask_1 = (results_mediapipe_1.segmentation_mask > 0.5) * 255
        mask_2 = (results_mediapipe_2.segmentation_mask > 0.5) * 255
        mask_3 = (results_mediapipe_3.segmentation_mask > 0.5) * 255

        img_0, mask_0 = self.crop_image(img_0, mask_0)
        
        img_0 = img_0 / 255
        mask_0 = mask_0 / 255
        img_0 = img_0 - [0.485, 0.456, 0.406]
        img_0 = img_0 / [0.229, 0.224, 0.225]
        img_0 = img_0.transpose(2, 0, 1)

        img_0[:, mask_0[:, :] == 0] = 0
        
        img_1, mask_1 = self.crop_image(img_1, mask_1)
        
        img_1 = img_1 / 255
        mask_1 = mask_1 / 255
        img_1 = img_1 - [0.485, 0.456, 0.406]
        img_1 = img_1 / [0.229, 0.224, 0.225]
        img_1 = img_1.transpose(2, 0, 1)

        img_1[:, mask_1[:, :] == 0] = 0
        
        img_2, mask_2 = self.crop_image(img_2, mask_2)
        
        img_2 = img_2 / 255
        mask_2 = mask_2 / 255
        img_2 = img_2 - [0.485, 0.456, 0.406]
        img_2 = img_2 / [0.229, 0.224, 0.225]
        img_2 = img_2.transpose(2, 0, 1)

        img_2[:, mask_2[:, :] == 0] = 0
        
        img_3, mask_3 = self.crop_image(img_3, mask_3)
        
        img_3 = img_3 / 255
        mask_3 = mask_3 / 255
        img_3 = img_3 - [0.485, 0.456, 0.406]
        img_3 = img_3 / [0.229, 0.224, 0.225]
        img_3 = img_3.transpose(2, 0, 1)

        img_3[:, mask_3[:, :] == 0] = 0
        
        
        imgtensor_0 = torch.FloatTensor(np.concatenate((img_0, mask_0[None]))).unsqueeze(0)
        imgtensor_1 = torch.FloatTensor(np.concatenate((img_1, mask_1[None]))).unsqueeze(0)
        imgtensor_2 = torch.FloatTensor(np.concatenate((img_2, mask_2[None]))).unsqueeze(0)
        imgtensor_3 = torch.FloatTensor(np.concatenate((img_3, mask_3[None]))).unsqueeze(0)
        
        #########
        
        img_icon_0, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_0.png', 'pymaf', 512)
        img_icon_1, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_1.png', 'pymaf', 512)
        img_icon_2, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_2.png', 'pymaf', 512)
        img_icon_3, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path + '_3.png', 'pymaf', 512)

        data_dict = {
            'name' : img_name, 
            'image_tensor' : imgtensor_0, 
            'image_tensor_0' : imgtensor_0, 
            'image_tensor_1' : imgtensor_1, 
            'image_tensor_2' : imgtensor_2, 
            'image_tensor_3' : imgtensor_3, 
            'image' : img_icon_0.unsqueeze(0), 
            'image_0' : img_icon_0.unsqueeze(0), 
            'image_1' : img_icon_1.unsqueeze(0), 
            'image_2' : img_icon_2.unsqueeze(0), 
            'image_3' : img_icon_3.unsqueeze(0)
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








