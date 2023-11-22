

import pandas as pd

import os
import shutil
import json

import pymeshlab as ml




old_path = './여자'
new_path = './woman'



data_info_dict = {}

for folder_name in os.listdir(old_path):
    folder_path = os.path.join(old_path, folder_name)
    if os.path.isdir(folder_path):
        
        for file_name in os.listdir(folder_path):
            
            file_sheet_name = file_name[:2] + 'SM_00' + file_name[2:-4]
            df = pd.read_excel(folder_path + '.xlsx', engine = 'openpyxl', sheet_name = file_sheet_name).fillna('0')
            
            gender = df.keys()[3][-6:-5]
            age = df.keys()[3][-2:]
            
            df[df.keys()[1]] = df[df.keys()[1]] + ' ' + df[df.keys()[2]]
            
            df = df.drop([df.keys()[0], df.keys()[2]], axis = 1)
            info_dict = df.set_index(df.keys()[0]).T.to_dict('list')
            
            data_info_dict[file_name] = {
                'gender' : gender, 
                'age' : age, 
                'info' : info_dict
            }

            file_name_path = os.path.join(folder_path, file_name)
            new_file_name_path = os.path.join(new_path, 'obj_data', file_name)
            
            ms = ml.MeshSet()
            ms.load_new_mesh(file_name_path)
            m = ms.current_mesh()

            #Target number of vertex
            TARGET = 25000

            #Estimate number of faces to have 100+10000 vertex using Euler
            numFaces = 100 + 2 * TARGET

            #Simplify the mesh. Only first simplification will be agressive
            while (ms.current_mesh().vertex_number() > TARGET):
                ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum = numFaces, preservenormal = True)
                #Refine our estimation to slowly converge to TARGET vertex number
                numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

            m = ms.current_mesh()
            ms.save_current_mesh(new_file_name_path)
            
            new_file_name_path_smpl = os.path.join(new_path, 'obj_data_smpl', file_name)
            
            ms = ml.MeshSet()
            ms.load_new_mesh(file_name_path)
            m = ms.current_mesh()

            #Target number of vertex
            TARGET = 6890

            #Estimate number of faces to have 100+10000 vertex using Euler
            numFaces = 13776

            #Simplify the mesh. Only first simplification will be agressive
            while (ms.current_mesh().vertex_number() > TARGET):
                ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum = numFaces, preservenormal = True)
                #Refine our estimation to slowly converge to TARGET vertex number
                numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

            m = ms.current_mesh()
            ms.save_current_mesh(new_file_name_path_smpl)

json_neme = os.path.join(new_path, 'datainfo.json')

with open(json_neme, 'w') as f:
    json.dump(data_info_dict, f, ensure_ascii = False, indent = 4)











































































