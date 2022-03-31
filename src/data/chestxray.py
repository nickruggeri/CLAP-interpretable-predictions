from typing import Any, Callable, List, Optional, Union

import PIL
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset

from .utils import DATA_DIR

ROOT_DIR = DATA_DIR / "chestxray" / "CXR8"
LABELS = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Effusion": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Hernia": 7,
    "Infiltration": 8,
    "Mass": 9,
    "No Finding": 10,
    "Nodule": 11,
    "Pleural_Thickening": 12,
    "Pneumonia": 13,
    "Pneumothorax": 14,
}


class ChestXRay(VisionDataset):
    def __init__(
        self,
        train: bool = True,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(str(ROOT_DIR), transforms, transform, target_transform)
        self.train = train

        self.filename, self.y = self._load_data()

    def _load_data(self) -> Union[List, np.ndarray]:
        idx_file = (
            ROOT_DIR / "train_val_list.txt"
            if self.train
            else ROOT_DIR / "test_list.txt"
        )
        with open(idx_file, "r") as file:
            images = set(map(lambda s: s.strip("\n"), file.readlines()))
        info_df = pd.read_csv(
            ROOT_DIR / "Data_Entry_2017_v2020.csv", index_col="Image Index"
        )

        # select only images specified in the train or test split
        info_df = info_df[info_df.index.isin(images)]
        filename = list(info_df.index)

        # extract labels
        info_df["Finding Labels"] = info_df["Finding Labels"].map(
            lambda label: label.split("|")
        )

        y = np.zeros((len(filename), len(LABELS)), dtype=np.int8)
        for i, (image_file, (index, row)) in enumerate(
            zip(filename, info_df.iterrows())
        ):
            assert index == image_file
            for disease in row["Finding Labels"]:
                y[i, LABELS[disease]] = 1

        return filename, y

    def __len__(self) -> int:
        return len(self.filename)

    def __getitem__(self, index: int) -> Any:
        y = self.y[index, ...]

        img_file = self.filename[index]
        x = PIL.Image.open(ROOT_DIR / "images" / "images" / img_file)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
