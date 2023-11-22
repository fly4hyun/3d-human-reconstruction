# training code (~ing)

dataset : https://sizekorea.kr/

'/woman' 폴더 내에
'/woman/obj_data', 
'/woman/obj_data_smpl', 
'/woman/img_data' 폴더 존재

원본 obj 파일 -> '/woman/obj_data'

data_cleaning.py를 통해 smpl 형식의 obj 생성 후 저장 -> '/woman/obj_data_smpl'

3d_to_img_gen.py, 3d_to_img_gen_etc.py 을 통해 obj를 각 방향에서 바라보는 이미지 저장 -> '/woman/img_data'

실루엣 기반 loss로 학습 진행 중 프로젝트 중단
