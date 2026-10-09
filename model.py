import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from pathlib import Path


# مسیر فایل مدل
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mobilenet_v3_small.pth"
)

# اطلاعات مدل
weights = MobileNet_V3_Small_Weights.DEFAULT

# ساخت مدل بدون دانلود
model = mobilenet_v3_small(weights=None)

# بارگذاری وزن‌های محلی
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# preprocessing
preprocess = weights.transforms()

# کلاس‌های ImageNet
categories = weights.meta["categories"]


def predict_image(image: Image.Image):

    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    probabilities = torch.nn.functional.softmax(
        output[0],
        dim=0
    )

    confidence, index = torch.max(
        probabilities,
        0
    )

    label = categories[index.item()]

    return label, confidence.item()