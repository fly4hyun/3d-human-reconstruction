


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

    render.load_meshes(torch_mesh_vert, mesh_face[0])
    img_0, _ = render.get_rgb_image()
    img_0 = ((img_0[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(torch_mesh_vert.gather(2, index_1) * torch.tensor([-1.0, 1.0, 1.0]), mesh_face)
    img_1, _ = render.get_rgb_image()
    img_1 = ((img_1[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    render.load_meshes(torch_mesh_vert * torch.tensor([-1.0, 1.0, -1.0]), mesh_face)
    img_2, _ = render.get_rgb_image()
    img_2 = ((img_2[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    

    
    render.load_meshes(torch_mesh_vert.gather(2, index_1) * torch.tensor([1.0, 1.0, -1.0]), mesh_face)
    img_3, _ = render.get_rgb_image()
    img_3 = ((img_3[0].permute(1, 2, 0) + 1.0) * 255.0 / 2.0).detach().cpu().numpy().astype(np.uint8)
    
    Image.fromarray(img_0).save(img_path + '_0.png')
    Image.fromarray(img_1).save(img_path + '_1.png')
    Image.fromarray(img_2).save(img_path + '_2.png')
    Image.fromarray(img_3).save(img_path + '_3.png')