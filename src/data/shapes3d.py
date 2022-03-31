"""
Shaeps3D data set.
https://github.com/deepmind/3d-shapes
"""
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from six.moves import range
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from .utils import DATA_DIR, GroundTruthData

SHAPES3D_PATH = DATA_DIR / "shapes3d" / "3dshapes.h5"


class Shapes3D(GroundTruthData):
    factor_to_idx = {
        "floor_hue": 0,
        "wall_hue": 1,
        "object_hue": 2,  # from 0 to 9
        "scale": 3,  # from 0 to 7
        "shape": 4,  # 0 = square, 1 = cylinder, 2 = sphere, 3 = oval
        "orientation": 5,
    }

    def __init__(self, transforms: Any) -> None:
        self.transforms = transforms

        self.images, self.factors = self._load_data()
        self.factor_values = [
            np.unique(self.factors[:, i]) for i in range(self.num_factors)
        ]
        self.factor_sizes = list(map(len, self.factor_values))
        self._factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)

    @property
    def num_factors(self) -> int:
        return self.factors.shape[1]

    @property
    def factors_num_values(self) -> List[np.ndarray]:
        return self.factor_values

    @property
    def observation_shape(self) -> int:
        return self.images.shape[1:]

    @property
    def factor_bases(self) -> np.ndarray:
        return self._factor_bases

    def sample_factors(self, num: int) -> np.ndarray:
        """Sample a batch of factors Y."""
        vals = self.factors_num_values
        sample = np.stack([np.random.choice(val, size=num) for val in vals], axis=1)
        assert sample.shape == (num, self.num_factors)
        return sample

    def sample_observations_from_factors(
        self,
        factors: np.ndarray,
        random_state: Optional[
            int
        ] = None,  # not used, here for parent class compatibility
        draw_label=False,
    ) -> Dict[str, np.ndarray]:
        """Sample a batch of observations X given a batch of factors Y."""
        inds = np.array(np.dot(factors, self.factor_bases), dtype=np.int64)
        n = len(factors)
        c, k = self.images[0].shape[2], self.images[0].shape[1]  # numpy array is WHC
        x = torch.empty(n, c, k, k)
        for i in range(n):
            x[i] = self.transforms(self.images[inds[i]])
        out_dict = {"x": x}
        if draw_label:
            labels = label_function(self.factors[inds])
            out_dict["y"] = torch.from_numpy(labels)
        return out_dict

    @staticmethod
    def _load_data():
        all_data = h5py.File(SHAPES3D_PATH, "r")
        # load h5py into numpy array, as h5py gives multiprocessing problems when used within torch Dataloaders
        images, factors = all_data["images"][...], all_data["labels"][...]

        # make labels integers
        floor_hue = factors[:, 0] * 10
        assert np.allclose(np.unique(floor_hue), np.array(range(10)))

        wall_hue = factors[:, 1] * 10
        assert np.allclose(np.unique(wall_hue), np.array(range(10)))

        object_hue = factors[:, 2] * 10
        assert np.allclose(np.unique(object_hue), np.array(range(10)))

        scale = LabelEncoder().fit_transform(factors[:, 3])
        assert np.all(np.unique(scale) == np.array(range(8)))

        shape = factors[:, 4]

        orientation = LabelEncoder().fit_transform(factors[:, 5])
        assert np.all(np.unique(orientation) == np.array(range(15)))

        integer_factors = np.stack(
            [floor_hue, wall_hue, object_hue, scale, shape, orientation], axis=1
        ).astype(np.uint8)

        return images, integer_factors

    def __len__(self) -> int:
        return len(self.images)


class Shapes3DDataset(Dataset):
    def __init__(
        self,
        ground_truth_data: Shapes3D,
        train: bool,
        test_size: float = 0.1,
        random_state: Optional[int] = 12345,
    ) -> None:
        self.ground_truth_data = ground_truth_data
        self.transforms = self.ground_truth_data.transforms

        self.train = train
        if train:
            self.ind, _ = train_test_split(
                range(len(self.ground_truth_data)),
                test_size=test_size,
                random_state=random_state,
            )
        else:
            _, self.ind = train_test_split(
                range(len(self.ground_truth_data)),
                test_size=test_size,
                random_state=random_state,
            )

        # check whether index is in allowed set ( train / test set)
        self.check_ind = torch.zeros(len(self.ground_truth_data))
        self.check_ind[self.ind] = 1

    def __len__(self) -> int:
        return len(self.ind)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """idx is unused"""
        while True:
            factors = self.ground_truth_data.sample_factors(1)[0]
            ind = np.array(
                np.dot(factors, self.ground_truth_data.factor_bases), dtype=np.int64
            )
            # check if image in train / test set
            if self.check_ind[ind] == 1:
                if self.transforms is not None:
                    x = self.transforms(self.ground_truth_data.images[ind])
                else:
                    x = self.ground_truth_data.images[ind]
                factors = self.ground_truth_data.factors[ind]
                label = label_function(factors)
                return x, label

    @property
    def factor_data(self) -> Shapes3D:
        return self.ground_truth_data


def label_function(factors: np.ndarray) -> np.ndarray:
    # define labels: different combinations of scale and color
    color, scale = factors[..., 2], factors[..., 3]

    label1 = (scale <= 5) & (color >= 3)
    label2 = (scale >= 3) & (color >= 3)
    label3 = (scale <= 4) & (color >= 2)
    label4 = scale >= 5

    label = np.stack([label1, label2, label3, label4], axis=-1)
    label = label.astype(np.uint8)

    return label
