# 3D Human Body Reconstruction from Single / 4-View Images

> 단일 또는 4방향 이미지만으로 3D 체형 복원 및 신체 치수 자동 측정 모델 개발
> 기간: 2022.09 – 2023.03

---

## Summary

* SMPL 기반 LVD + ICON 결합으로 **의복 상태에서도 체형 복원 가능한 3D 모델** 구현
* 복원 결과로부터 **부위별 치수를 자동 산출하는 측정 알고리즘**까지 일관 파이프라인 구축

---

## My Contribution

* **LVD + ICON 결합 구조 구현 및 튜닝**으로 복원 성능 개선
* trimesh 기반 body_measurements 라이브러리 수정 → **자동 치수 계산 알고리즘 설계**
* 치수 정확도 검증(10명 기준 평균 오차율 7%) 및 실사용 수준 검정

---

## Performance (10명 기준 평균 절대 오차율)

| 항목   | 오차율 |
| ---- | --- |
| 몸무게  | 8%  |
| 목둘레  | 13% |
| 가슴둘레 | 4%  |
| 복부   | 6%  |
| 팔둘레  | 9%  |
| 엉덩이  | 3%  |
| 허벅지  | 5%  |

> 전체 평균 오차율: **7%**

---

## Deployment / Output

* 국제의료기기병원설비 전시회 **KIMES 2023 시연 수행**
* 체형예측시스템 특허 출원 (10-2023-0024467)

---

## Visual Examples

실증 이미지는 공개 가능 범위 내에서 별도 폴더에 예시 업로드 예정 (`docs/figures/`)

---

## Disclosure

* 시연 영상/자료는 외부 공개 가능 범위만 첨부 (민감 데이터 없음)

---

## References

* LVD: [https://github.com/enriccorona/LVD](https://github.com/enriccorona/LVD)
* ICON: [https://github.com/YuliangXiu/ICON](https://github.com/YuliangXiu/ICON)

## Model & Data (공개 가능한 범위)

* Model weights (예시 위치 표기만): `train_silhouette/model_save/`, `3d_api/model_save/`
* Segmentation weights: `train_silhouette/models_seg/`
* Demo dataset (woman): `train_silhouette/woman/`

> *실제 파일은 공유 링크/별도 전달로 관리 — 본 저장소에는 경로/구조만 표기*

---

## Contact

이현희 / AI Research Engineer
f
[ly4hyun@naver.com](mailto:ly4hyun@naver.com)
