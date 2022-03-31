import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import PIL
import numpy as np
from torchvision.datasets import VisionDataset

from .utils import DATA_DIR

ROOT_DIR = os.path.join(
    DATA_DIR, "leaf_disease", "Plant_leave_diseases_dataset_without_augmentation"
)
LEAF_TYPES = (
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper,_bell",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
)
DISEASE = {
    "Apple": ("Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"),
    "Blueberry": ("healthy",),
    "Cherry": ("Powdery_mildew", "healthy"),
    "Corn": (
        "Northern_Leaf_Blight",
        "Common_rust",
        "Cercospora_leaf_spot Gray_leaf_spot",
        "healthy",
    ),
    "Grape": (
        "Black_rot",
        "Esca_(Black_Measles)",
        "Leaf_blight_(Isariopsis_Leaf_Spot)",
        "healthy",
    ),
    "Orange": ("Haunglongbing_(Citrus_greening)",),
    "Peach": ("Bacterial_spot", "healthy"),
    "Pepper,_bell": ("Bacterial_spot", "healthy"),
    "Potato": ("Early_blight", "Late_blight", "healthy"),
    "Raspberry": ("healthy",),
    "Soybean": ("healthy",),
    "Squash": ("Powdery_mildew",),
    "Strawberry": ("Leaf_scorch", "healthy"),
    "Tomato": (
        "Bacterial_spot",
        "Early_blight",
        "Late_blight",
        "Leaf_Mold",
        "Septoria_leaf_spot",
        "Spider_mites Two-spotted_spider_mite",
        "Target_Spot",
        "Tomato_mosaic_virus",
        "Tomato_Yellow_Leaf_Curl_Virus",
        "healthy",
    ),
}


class PlantVillageDataset(VisionDataset):
    def __init__(
        self,
        leaf_type: str,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            ROOT_DIR, transform=transforms, target_transform=target_transform
        )

        assert (
            leaf_type in LEAF_TYPES
        ), f"leaf_type unknown. Choose one in \n{LEAF_TYPES}."
        self.leaf_type = leaf_type
        self._n_labels = len(DISEASE[self.leaf_type])

        self.filename, self.y = self._load_data()

    def _load_data(self) -> Tuple[List[str], np.ndarray]:
        filename = []
        y = []
        for disease_idx, disease in enumerate(DISEASE[self.leaf_type]):
            disease_dir = Path(ROOT_DIR) / (self.leaf_type + "___" + disease)
            for img_file in disease_dir.iterdir():
                filename.append(str(img_file))

                # store label
                label = np.zeros(self._n_labels, dtype=int)
                label[disease_idx] = 1
                y.append(label)

        return filename, np.stack(y)

    def __len__(self) -> int:
        return len(self.filename)

    def __getitem__(self, index: int) -> Any:
        y = self.y[index, ...]
        x = PIL.Image.open(self.filename[index])

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
