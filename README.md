# Capstone-Recycling_Resource_Recognition_Model
2023년 졸업작품 - 재활용 자원 이미지 인식 모델 (2023.02.01 ~ 2023.10.27)

## 프로젝트 개요

이 프로젝트는 2023년 졸업작품 재활용 자원 거래 선순환 플랫폼 앱에 탑재된 이미지 인식 모델입니다.

## 주요 기능

- 이미지에서 재활용 가능한 자원('나무류', '종이류', '플라스틱류', '스티로폼류', '페트병류', '캔류', '유리병류', '의류', '비닐류') 인식
- 자원 분류 및 분석
- 실시간 예측 및 결과 시각화

## 사용 언어 및 모델 프레임워크

- 언어: Python
- 모델 프레임워크: TensorFlow
  
  ![image](https://github.com/Kimdeokryun/Capstone-Recycling_Resource_Recognition_Model/assets/96904134/0dd71063-0f45-4361-a9c5-a4cbddee2af1)
- 사용 모델: EfficientNet B0  (224,224)


동일 파라미터 수 대비 Imagenet 데이터셋의 Top 1 Accuracy가 높은 것을 알 수 있다.

### 모델 선정
- 파라미터 수가 적어 모델의 크기가 적다.
- 모델의 크기가 적어야 tflite 변환 후 앱 어플리케이션 탑재 용량이 적어진다.
- 앱 어플리케이션의 부하가 적어진다.
- 정확도가 높은 편이다.
- Yolo와 다른 점은 단일 객체 인식 모델

## 사용 데이터셋
- AI HUB 의 생활 폐기물 이미지 데이터셋
- https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=140
- 대분류 데이터셋으로는 총 15개의 클래스로 존재
- 약 75만장의 이미지, 해당 모델 제작에서는 클래스의 수를 축소시켜 약 30만장의 이미지로 훈련

### 이미지 전처리
- 데이터셋 내의 label.xlsx 엑셀파일을 활용하여 train, val 라벨 별로 폴더 및 이미지 파일 존재
- OpenCV_Crop_Resize.py 를 통해서 이미지 데이터셋들을 1:1 크기로 Crop 및 Resize
- 활용할 데이터셋만 라벨로 분류하여 전처리 데이터셋으로 제작


## 모델 훈련 및 변환
- EfficientNet_optimize.py 를 통해 전처리 데이터셋을 EfficientNet B0 아키텍처를 사용한 모델로 훈련
- cpu_info.py 를 통해 cpu 및 gpu 모델로드 사용
- tensorflow를 사용하기 위해 CUDA 및 GPU 관련 파일 설치 및 환경 변수 설정
- predict_model.py 를 통해 test 데이터셋 정확도 측정
- Tflite 변환.py 를 통해 모델 tflite 확장자로 변환
- predict_tflite.py, predict_tflitemodel.py 를 통해 입력, 출력 텐서 확인 및 tflite 모델 정확도 측정
- 해당 데이터셋을 통해 batch_size 64 기준 1 Epoch당 약 50분 소요


### 모델 훈련 환경
- Window 11
- Ram 32GB
- GPU RTX3060
- IDE Pycharm, Python 3.9 
- CUDA version 11.2
- cuDNN v8.9.1
