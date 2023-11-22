

import glob

import cv2
import tqdm
import numpy as np
from copy import copy
from PIL import Image
import io

import pywavefront
from body_measurements.measurement import Body3D

import test_code_for_app


def measure():

    name = 'app_img'
    jf_text1 = 169
    test_code_for_app.test()

    lvd_person = pywavefront.Wavefront(
        "/jf-training-home/src/app_result/1_LVD_%s.obj"%name, 
        create_materials=True,
        collect_faces=True
    )
    #"/jf-training-home/src/app_result/3_ICON_%s_remesh.obj"%name, 
    #"/jf-training-home/src/app_result/2_ICON_%s_recon.obj"%name, 
    #"/jf-training-home/src/app_result/1_LVD_%s.obj"%name, 

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





    lvd_height = round(lvd_height, 2)
    lvd_weight = round(lvd_weight, 2)
    lvd_neck_length = round(lvd_neck_length, 2)
    lvd_bicep_length = round(lvd_bicep_length, 2)
    lvd_chest_length = round(lvd_chest_length, 2)
    lvd_waist_length = round(lvd_waist_length, 2)
    lvd_hip_length = round(lvd_hip_length, 2)
    lvd_thigh_length = round(lvd_thigh_length, 2)
    lvd_BMI = round(lvd_BMI, 2)






    ################################

    """

    icon_person = pywavefront.Wavefront(
        '/jf-training-home/src/app_result/ICON_0_remesh_%s.obj'%name, 
        create_materials=True,
        collect_faces=True
    )

    icon_faces = np.array(icon_person.mesh_list[0].faces)
    icon_vertices = np.array(icon_person.vertices) / 100

    icon_body = Body3D(icon_vertices, icon_faces)

    icon_body_measurements = icon_body.getMeasurements()

    icon_height = icon_body.height()
    icon_weight = icon_body.weight()
    _, _, icon_shoulder_length = icon_body.shoulder()
    _, _, icon_chest_length = icon_body.chest()
    _, _, icon_hip_length = icon_body.hip()
    _, _, icon_waist_length = icon_body.waist()
    _, _, icon_thigh_length = icon_body.thighOutline()
    icon_outer_leg_length = icon_body.outerLeg()
    icon_inner_leg_length = icon_body.innerLeg()
    _, _, icon_neck_length = icon_body.neck()
    icon_neck_hip_length = icon_body.neckToHip()

    icon_ratio = copy(float(jf_text1) / icon_height)
    icon_height = icon_height * icon_ratio
    icon_weight = icon_weight * icon_ratio * icon_ratio / 10000
    icon_shoulder_length = icon_shoulder_length * icon_ratio
    icon_chest_length = icon_chest_length * icon_ratio
    icon_hip_length = icon_hip_length * icon_ratio
    icon_waist_length = icon_waist_length * icon_ratio
    icon_thigh_length = icon_thigh_length * icon_ratio
    icon_outer_leg_length = icon_outer_leg_length * icon_ratio
    icon_inner_leg_length = icon_inner_leg_length * icon_ratio
    icon_neck_length = icon_neck_length * icon_ratio
    icon_neck_hip_length = icon_neck_hip_length * icon_ratio


    icon_height = round(icon_height, 2)
    icon_weight = round(icon_weight, 2)
    icon_shoulder_length = round(icon_shoulder_length, 2)
    icon_chest_length = round(icon_chest_length, 2)
    icon_hip_length = round(icon_hip_length, 2)
    icon_waist_length = round(icon_waist_length, 2)
    icon_thigh_length = round(icon_thigh_length, 2)
    icon_outer_leg_length = round(icon_outer_leg_length, 2)
    icon_inner_leg_length = round(icon_inner_leg_length, 2)
    icon_neck_length = round(icon_neck_length, 2)
    icon_neck_hip_length = round(icon_neck_hip_length, 2)

    """









    """
    Output 자동 생성 영역 (OUTPUT Category)
    """

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
            }
        ]
    }



    """

    output = {
        "text": [
            {
                "키": lvd_height,
                "몸무게": lvd_weight,
                "어깨 둘레": lvd_shoulder_length,
                "가슴 둘레": lvd_chest_length,
                "엉덩이 둘레": lvd_hip_length,
                "배 둘레": lvd_waist_length,
                "허벅지 둘레": lvd_thigh_length,
                "바깥쪽 다리 길이": lvd_outer_leg_length,
                "안쪽 다리 길이": lvd_inner_leg_length,
                "목 둘레": lvd_neck_length,
                "상체 길이": lvd_neck_hip_length,
            }, 
            {
                "키": icon_height,
                "몸무게": icon_weight,
                "어깨 둘레": icon_shoulder_length,
                "가슴 둘레": icon_chest_length,
                "엉덩이 둘레": icon_hip_length,
                "배 둘레": icon_waist_length,
                "허벅지 둘레": icon_thigh_length,
                "바깥쪽 다리 길이": icon_outer_leg_length,
                "안쪽 다리 길이": icon_inner_leg_length,
                "목 둘레": icon_neck_length,
                "상체 길이": icon_neck_hip_length,
            }
        ]
    }

    """

    print(output)

if __name__ == "__main__":
    
    measure()