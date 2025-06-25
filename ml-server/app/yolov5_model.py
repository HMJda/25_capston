from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def load_yolo(model_path):
    model = YOLO(model_path)
    return model

def detect_leaf(model, image):
    # PIL 이미지를 OpenCV 형식(BGR)으로 변환
    image_np = np.array(image)[:, :, ::-1].copy()  # RGB → BGR
    
    # YOLO 추론
    results = model(image_np)[0]
    
    # 잎 영역 크롭
    crops = []
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        crop = image.crop((x1, y1, x2, y2))  # PIL 형식으로 반환
        crops.append(crop)
    
    return crops
