import argparse
import numpy as np
from PIL import Image
import torch
from models.classifier import MelanomaClassifier
from data.augmentations import get_val_transforms
from utils.tta import tta_predict


def predict(image_path: str, checkpoint: str, use_tta: bool = True) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MelanomaClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    transforms = get_val_transforms()
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transforms(image=image)["image"].unsqueeze(0)

    if use_tta:
        prob = tta_predict(model, tensor.squeeze(0), device)
    else:
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor.to(device))).item()

    label = "MELANOMA" if prob >= 0.5 else "Benign"
    return {"probability": round(prob, 4), "prediction": label}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", default="models/fold1_best.pth")
    parser.add_argument("--no-tta",     action="store_true")
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint, use_tta=not args.no_tta)
    print(f"Prediction: {result['prediction']}  (p={result['probability']:.4f})")
