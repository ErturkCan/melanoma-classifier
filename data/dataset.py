import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None, has_labels: bool = True):
        self.df         = df.reset_index(drop=True)
        self.image_dir  = image_dir
        self.transform  = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row['image_name']}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        if self.has_labels:
            label = torch.tensor(row["target"], dtype=torch.float32)
            return image, label
        return image


import torch  # noqa: E402 — needed after class definition
