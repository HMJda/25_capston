import tensorflow as tf
import numpy as np
from PIL import Image

def load_binary_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_binary(image):
    # EfficientNet 전처리 적용
    resized = image.resize((260, 260))
    image_np = np.array(resized)
    return tf.keras.applications.efficientnet.preprocess_input(image_np)

def predict_binary(model, image):
    processed_image = preprocess_binary(image)
    input_tensor = np.expand_dims(processed_image, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]
    return pred[0] > 0.5  # True: 질병 있음
