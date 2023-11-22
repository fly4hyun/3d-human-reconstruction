"""
시스템 정보
아래 스크립트 삭제시 JF deploy 실행이 안됩니다.
#JF_DEPLOYMENT_INPUT_DATA_INFO_START
{
    "deployment_input_data_form_list": [
        {
            "method": "POST",
            "location": "file",
            "api_key": "image",
            "value_type": "file",
            "category": "image",
            "category_description": "png \uc0ac\uc9c4"
        },
        {
            "method": "POST",
            "location": "form",
            "api_key": "height",
            "value_type": "int",
            "category": "text",
            "category_description": "\ud0a4"
        }
    ]
}
#JF_DEPLOYMENT_INPUT_DATA_INFO_END
"""

import os

import sys
sys.path.append('/addlib')
from deployment_api_deco import api_monitor
from flask import Flask, request, jsonify
from flask.views import MethodView
from flask_cors import CORS
import argparse
import requests
import base64
parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='/')
"""
배포 실행 명령어 관련 자동생성 영역
"""

"""
사용자 추가 영역
"""

import glob
from datetime import datetime
import cv2
import tqdm
import numpy as np
from copy import copy
from PIL import Image
import io
import trimesh

import pywavefront
from body_measurements.measurement import Body3D

import test_code_for_app




params, _ = parser.parse_known_args()
params = vars(params)
app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})

class run_api(MethodView):
    def __init__(self):
        pass

    @api_monitor()
    def get(self):
        return "JF DEPLOYMENT RUNNING"

    @api_monitor()
    def post(self):
        """
        TEST API 받아오는 부분 자동 생성 영역
        """
        jf_text1 = request.form['height']
        jf_image1_files = request.files.getlist('image')
        jf_image1 = request.files['image']
        
        
        """
        사용자 영역
        # 필요한 전처리
        # 배포 테스트에 필요한 처리 등 
        """
        now = datetime.now()
        name = str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_' + str(now.second) + '_img'
        
        image_bytes = jf_image1.read()
        img = Image.open(io.BytesIO(image_bytes))
        img.save("./app_data/%s.png"%name, "PNG")
        
        
        test_code_for_app.test(name)
        
        lvd_person = pywavefront.Wavefront(
            "./app_result/1_LVD_%s.obj"%name, 
            create_materials=True,
            collect_faces=True
        )

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
        
        if lvd_BMI <= 18.5:
            BMI_result = '저체중'
        elif lvd_BMI < 23.0:
            BMI_result = '정상'
        elif lvd_BMI < 25.0:
            BMI_result = '과체중'
        else:
            BMI_result  = '비만'

        lvd_height = round(lvd_height, 2)
        lvd_weight = round(lvd_weight, 2)
        lvd_neck_length = round(lvd_neck_length, 2)
        lvd_bicep_length = round(lvd_bicep_length, 2)
        lvd_chest_length = round(lvd_chest_length, 2)
        lvd_waist_length = round(lvd_waist_length, 2)
        lvd_hip_length = round(lvd_hip_length, 2)
        lvd_thigh_length = round(lvd_thigh_length, 2)
        lvd_BMI = round(lvd_BMI, 2)

        
        mesh = open("./app_result/4_ICON_%s_refine.obj"%name, 'r')
        mesh_txt = mesh.read()
        mesh.close()


        ################################





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
                    "비만 결과" : BMI_result,
                }
            ], 
            "obj": [
                {
                    "3d_human": {"obj": mesh_txt}
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
        
        os.system("rm ./app_data/*")
        os.system("rm ./app_result/*")
        
        
        return jsonify(output)

app.add_url_rule(params["prefix"], view_func=run_api.as_view("run_api"))
if __name__ == "__main__":
    """
    모델 로드를 권장하는 위치
    사용자 영역
    """
    #model_run = test_code_for_app()
    app.run('0.0.0.0',8555,threaded=True)