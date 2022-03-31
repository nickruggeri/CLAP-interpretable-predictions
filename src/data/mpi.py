from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import DATA_DIR, GroundTruthData, SplitDiscreteStateSpace


class MPIData(GroundTruthData):
    def __init__(self, transforms: Any, save_labels_and_factors: bool = False) -> None:
        mpi3d_path = DATA_DIR / "mpidata" / "mpi3d_real.npz"
        mpi3d_path_raw = DATA_DIR / "mpidata" / "mpi3d_real_raw.npz"
        if save_labels_and_factors:
            data = np.load(mpi3d_path_raw)
        else:
            data = np.load(mpi3d_path)
        self.images = data["images"]
        self.transforms = transforms
        self.ind = np.arange(len(self.images))
        self._factor_sizes = [
            6,
            6,
            2,
            3,
            3,
            40,
            40,
        ]  # object color, shape, size, height, background color, horizontal axis, vertical axis
        self._num_factors = len(self.factor_sizes)
        self._num_factors_core = 3
        self._num_factors_style = 4
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes)
        self._factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        if save_labels_and_factors:
            self.label_fct = label_fct
            self.create_labels_and_factors()
            np.savez(
                mpi3d_path, images=self.images, labels=self.labels, factors=self.factors
            )
        else:
            self.labels = data["labels"]
            self.factors = data["factors"]
            self.a = torch.tensor(
                self.factors_to_domain(self.factors), dtype=torch.float
            )

    @property
    def factor_sizes(self) -> List[int]:
        return self._factor_sizes

    @property
    def num_factors(self) -> int:
        return self._num_factors

    @property
    def num_factors_core(self) -> int:
        return self._num_factors_core

    @property
    def num_factors_style(self) -> int:
        return self._num_factors_style

    @property
    def factor_bases(self) -> np.ndarray:
        return self._factor_bases

    def sample_factors(self, num: int) -> np.ndarray:
        return self.state_space.sample_latent_factors(num)

    def sample_observations_from_factors(
        self,
        factors: np.ndarray,
        random_state: Optional[
            int
        ] = None,  # not used, here for parent class compatibility
        draw_label: bool = False,
    ) -> Dict[str, np.ndarray]:
        inds = np.array(np.dot(factors, self.factor_bases), dtype=np.int64)
        n = len(factors)
        c, k = self.images[0].shape[2], self.images[0].shape[1]  # numpy array is WHC
        x = torch.empty(n, c, k, k)
        for i in range(n):
            x[i] = self.transforms(self.images[inds[i]])
        out_dict = {"x": x}
        if draw_label:
            out_dict["y"] = torch.from_numpy(self.labels[inds])
        return out_dict

    def create_labels_and_factors(self) -> None:
        n = len(self.images)
        labels = np.empty(shape=(n, 4))
        factors = np.empty(shape=(n, self.num_factors))
        for i in tqdm(range(n)):
            ind = self.ind[i]
            i_factors = self.idx_to_factors(ind)
            factors[i, :] = i_factors
            labels[i, :] = self.label_fct(i_factors)
        self.factors = factors
        self.labels = np.float32(labels)

    @staticmethod
    def factors_to_domain(factors: np.ndarray) -> np.ndarray:
        return factors // 10


class MPIDataset(Dataset):
    def __init__(
        self,
        mpi_data: MPIData,
        train: bool,
        test_size: float = 0.4,
        random_state: Optional[int] = 12345,
    ) -> None:
        self.train = train
        self.mpi_data = mpi_data
        self.transforms = self.mpi_data.transforms
        if train:
            self.ind, _ = train_test_split(
                range(len(mpi_data.images)),
                test_size=test_size,
                random_state=random_state,
            )
        else:
            _, self.ind = train_test_split(
                range(len(mpi_data.images)),
                test_size=test_size,
                random_state=random_state,
            )

        self.n = len(self.ind)
        # check whether index is in allowed set ( train / test set)
        self.check_ind = torch.zeros(len(self.mpi_data.images))
        self.check_ind[self.ind] = 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:

        while True:
            factors = self.mpi_data.sample_factors(1)[0]
            ind = np.array(np.dot(factors, self.mpi_data.factor_bases), dtype=np.int64)
            # check if image in train / test set
            if self.check_ind[ind] == 1:
                x = self.transforms(self.mpi_data.images[ind])
                label = self.mpi_data.labels[ind]
                return x, label

    @property
    def factor_data(self) -> MPIData:
        return self.mpi_data


def label_fct(factors: np.ndarray) -> np.ndarray:
    y = np.empty(shape=(4,), dtype=np.int_)

    # first label
    if factors[0] in [0, 1, 4, 5] and factors[1] in [0, 1, 2, 5] and factors[2] in [0]:
        y[0] = 1
    else:
        y[0] = 0

        # second label
        if factors[2] in [1] and factors[0] in [1, 2, 3]:
            y[1] = 1
        else:
            y[1] = 0

    # third label
    if factors[1] in [0, 4]:
        y[2] = 1
    else:
        y[2] = 0

    # fourth label
    if factors[1] in [2, 3, 4] and factors[2] in [1]:
        y[3] = 1
    else:
        y[3] = 0
    return y
