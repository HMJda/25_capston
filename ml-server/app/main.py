from fastapi import FastAPI, UploadFile, File
from app.yolov5_model import load_yolo, detect_leaf
from app.binary_model import load_binary_model, predict_binary
from app.multiclass_model import load_multiclass_model, preprocess_multiclass, predict_multiclass
from PIL import Image
import io
import torch

app = FastAPI()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
yolo_model = load_yolo("models/best.pt")
binary_model = load_binary_model("models/best_efficientnetb2_binary_crop.keras")
multi_model = load_multiclass_model("models/best_model.pt", device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # YOLO로 잎 영역 추출
    cropped_leaves = detect_leaf(yolo_model, image)

    results = []
    for leaf in cropped_leaves:
        is_diseased = predict_binary(binary_model, leaf)
        
        if is_diseased:
            # PyTorch 전용 전처리
            leaf_tensor = preprocess_multiclass(leaf).to(device)
            disease_type = predict_multiclass(multi_model, leaf_tensor)
        else:
            disease_type = "healthy"
        
        results.append(disease_type)

    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
