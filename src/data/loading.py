from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from .chestxray import ChestXRay
from .mpi import MPIData, MPIDataset
from .plant_village import PlantVillageDataset
from .shapes3d import Shapes3D, Shapes3DDataset
from .small_norb import SmallNORB, SmallNORBDataset

dataset_statistics = {
    # dataset   # mean           # std
    "MPI": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    "SmallNORB": [[0.0], [1.0]],
    "Shapes3D": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    "PlantVillage": [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    "ChestXRay": [[0.0], [1.0]],
}


# Remark: while we use the following transformations at loading time, preprocessing some datasets yields a big speedup
# in terms of training time.
def vanilla_transform(dataset_name: str) -> Compose:
    return Compose(
        [
            ToTensor(),
            Normalize(*dataset_statistics[dataset_name]),
        ]
    )


def resize_transform(dataset_name: str, in_dim: int, n_channels=3) -> Compose:
    size = (in_dim, in_dim) if n_channels == 3 else in_dim
    return Compose(
        [
            ToTensor(),
            Resize(size=size),
            Normalize(*dataset_statistics[dataset_name]),
        ]
    )


def get_datasets(dataset_name: str) -> Tuple[Dataset, Dataset, int, int, int]:
    if dataset_name == "MPI":
        transforms = vanilla_transform(dataset_name)
        factor_data = MPIData(
            transforms=transforms,
            save_labels_and_factors=False,
        )
        train_dataset = MPIDataset(factor_data, train=True)
        test_dataset = MPIDataset(factor_data, train=False)
        n_channels, image_dim, n_classes = 3, 64, 4
    elif dataset_name == "SmallNORB":
        transforms = vanilla_transform(dataset_name)
        factor_data = SmallNORB(transforms)
        train_dataset = SmallNORBDataset(factor_data, train=True)
        test_dataset = SmallNORBDataset(factor_data, train=False)
        n_channels, image_dim, n_classes = 1, 64, 4
    elif dataset_name == "Shapes3D":
        transforms = resize_transform(dataset_name, 64, 3)
        factor_data = Shapes3D(transforms)
        train_dataset = Shapes3DDataset(
            train=True, ground_truth_data=factor_data, test_size=0.4
        )
        test_dataset = Shapes3DDataset(
            train=False, ground_truth_data=factor_data, test_size=0.4
        )
        n_channels, image_dim, n_classes = 3, 64, 4
    elif dataset_name == "PlantVillage":
        transforms = resize_transform(dataset_name, 64)
        dataset = PlantVillageDataset(
            "Tomato",
            transforms=transforms,
        )
        train_idx, test_idx = train_test_split(
            np.arange(len(dataset)), train_size=0.9, random_state=1234
        )
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        n_channels, image_dim, n_classes = 3, 64, 10
    elif dataset_name == "ChestXRay":
        transforms = Compose(
            [
                # some images have 1 channel, others 4. Make them all 1 channel.
                Grayscale(num_output_channels=1),
                resize_transform(dataset_name, 64),
            ]
        )
        train_dataset = ChestXRay(
            train=True,
            transforms=transforms,
        )
        test_dataset = ChestXRay(
            train=False,
            transforms=transforms,
        )
        n_channels, image_dim, n_classes = 3, 64, 15
    else:
        raise NotImplementedError(f"Dataset {dataset_name} unknown.")

    # store mean and variance as dataset attributes. Ugly but needed.
    # Same with factor indices.
    for dataset in (train_dataset, test_dataset):
        setattr(dataset, "mean", dataset_statistics[dataset_name][0])
        setattr(dataset, "std", dataset_statistics[dataset_name][1])

    return train_dataset, test_dataset, n_channels, image_dim, n_classes
