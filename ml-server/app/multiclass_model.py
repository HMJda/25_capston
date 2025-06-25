import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

CLASS_NAMES = ['anthracnose', 'leaf_spot', 'mixed', 'virus']

class EfficientNetModel(nn.Module):
    def __init__(self, num_diseases=4, pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_diseases)
        )
    def forward(self, x):
        features = self.backbone(x)
        disease_logits = self.classifier(features)
        return disease_logits

def load_multiclass_model(model_path, device):
    model = EfficientNetModel(num_diseases=4, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_multiclass(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(412),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_multiclass(model: nn.Module, image_tensor: torch.Tensor) -> str:
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        class_idx = torch.argmax(probs).item()
    return CLASS_NAMES[class_idx]
