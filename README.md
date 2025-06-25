### 업데이트된 README 및 요구사항 파일  

#### 1. 프로젝트 개요  
**25_capston**은 YOLO/efficientnet_b4 기반 객체 감지와 분류 작업을 수행하는 ML 프로젝트로,  
- **호랑이/잎사귀 패턴 감지**에 특화  
- **소프트맥스 분류** 적용

#### 2. 개발 환경  
```markdown
- **OS**: Windows 11  
- **CPU**: AMD Ryzen 7 5800X  
- **GPU**: NVIDIA RTX 3070 TI  
- **CUDA**: 11.8.0  
- **cuDNN**: 8.6.0  
- **Python**: 3.10  
```

#### 3. 설치 가이드  
**requirements.txt** 생성:  
```text
numpy==1.23.5
tensorflow==2.10.0
matplotlib==3.10.1
seaborn==0.13.2
scikit-learn==1.6.1
pillow==11.2.1
tqdm==4.67.1
opencv-python==4.11.0.86
pandas==2.2.3
```
> 실행 명령: `pip install -r requirements.txt`
```
```
#### 4. 프로젝트 구조  
```plaintext
25_capston/
├── ML/                  # 모델 및 분류 코드
│   ├── cropdataset/     # 원본 데이터
│   └── results
│         ├── b4_results_0625_4class_softmax  # 모델 코드
│         └── dataset_crop_0625_4class/ # train/val/test 분리한 학습용 데이터셋
├── ml-server/           # Docker 배포용 서버
└── requirements.txt     # 의존성 파일
```


### 요구사항 파일 생성  
**requirements.txt**  
```text
numpy==1.23.5
tensorflow==2.10.0
matplotlib==3.10.1
seaborn==0.13.2
scikit-learn==1.6.1
pillow==11.2.1
tqdm==4.67.1
opencv-python==4.11.0.86
pandas==2.2.3
```
> 파일 위치: 프로젝트 루트 디렉토리에 저장

