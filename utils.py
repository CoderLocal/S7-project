# app/utils.py

import numpy as np
import cv2
from typing import Dict

# Preprocess Image
def preprocess_image(image_bytes: bytes):
    # Convert byte data to image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))  # Resize to 256x256
    img = img.transpose((2, 0, 1))  # Convert to channel-first
#    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize
    return img

# Preprocess Tabular Data
def preprocess_tabular(data: Dict):
    # Assuming all features are numeric and scaled
    # Convert tabular data dictionary into a tensor
 #   tabular_features = torch.tensor([list(data.values())], dtype=torch.float32)
    tabular_features = {}
    return tabular_features

