


import os
import numpy as np
from PIL import Image

import torch
import trimesh


from renderlib.render import Render


device = torch.device('cuda:0')
render = Render(size = 512, device = device)



load_path = './woman/obj_data'
save_path = './woman/img_data'


for obj_name in os.listdir(load_path):

    obj_path = os.path.join(load_path, obj_name)
    img_path = os.path.join(save_path, obj_name[:-4])
    
    m = trimesh.load(obj_path)
    
    mesh_face = torch.tensor(m.faces).unsqueeze(0)
    torch_mesh_vert = torch.FloatTensor(m.vertices).unsqueeze(0)

    torch_mesh_vert = 1.8 * ((torch_mesh_vert - torch_mesh_vert[0, :, 1].min()) / (torch_mesh_vert[0, :, 1].max() - torch_mesh_vert[0, :, 1].min())) - 0.89

    torch_mesh_vert[0, :, 0] = torch_mesh_vert[0, :, 0] - torch_mesh_vert[0, :, 0].mean()
    torch_mesh_vert[0, :, 2] = torch_mesh_vert[0, :, 2] - torch_mesh_vert[0, :, 2].mean()


    index_1 = torch.tensor([2, 1, 0]).unsqueeze(0).unsqueeze(0)
    index_1 = index_1.expand(torch_mesh_vert.size(0), torch_mesh_vert.size(1), index_1.size(2))
    
    index_2 = torch.tensor([0, 2, 1]).unsqueeze(0).unsqueeze(0)
    index_2 = index_2.expand(torch_mesh_vert.size(0), torch_mesh_vert.size(1), index_1.size(2))

    
    temp = torch.cat([(torch_mesh_vert[:, :, 0] - torch_mesh_vert[:, :, 2]).unsqueeze(-1)  / 2 ** 0.5, torch_mesh_vert[:, :, 1].unsqueeze(-1), (torch_mesh_vert[:, :, 0] + torch_mesh_vert[:, :, 2]).unsqueeze(-1)  / 2 ** 0.5], dim = -1)

    
    render.load_meshes(temp, mesh_face[0])
    img_4, _ = render.get_rgb_image()
    img_4 = ((img_4[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(temp.gather(2, index_1) * torch.tensor([-1.0, 1.0, 1.0]), mesh_face)
    img_5, _ = render.get_rgb_image()
    img_5 = ((img_5[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(temp * torch.tensor([-1.0, 1.0, -1.0]), mesh_face)
    img_6, _ = render.get_rgb_image()
    img_6 = ((img_6[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(temp.gather(2, index_1) * torch.tensor([1.0, 1.0, -1.0]), mesh_face)
    img_7, _ = render.get_rgb_image()
    img_7 = ((img_7[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(torch_mesh_vert.gather(2, index_2) * torch.tensor([1.0, -1.0, 1.0]), mesh_face[0])
    img_8, _ = render.get_rgb_image()
    img_8 = ((img_8[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(torch_mesh_vert.gather(2, index_2) * torch.tensor([1.0, 1.0, -1.0]), mesh_face[0])
    img_9, _ = render.get_rgb_image()
    img_9 = ((img_9[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    Image.fromarray(img_4).save(img_path + '_4.png')
    Image.fromarray(img_5).save(img_path + '_5.png')
    Image.fromarray(img_6).save(img_path + '_6.png')
    Image.fromarray(img_7).save(img_path + '_7.png')
    Image.fromarray(img_8).save(img_path + '_8.png')
    Image.fromarray(img_9).save(img_path + '_9.png')