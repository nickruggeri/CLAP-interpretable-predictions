# from
# https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/norb.py

"""SmallNORB dataset."""
from __future__ import absolute_import, division, print_function

import os

import PIL
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from six.moves import range
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import (
    DATA_DIR,
    GroundTruthData,
    SplitDiscreteStateSpace,
    StateSpaceAtomIndex,
)

SMALLNORB_TEMPLATE = os.path.join(DATA_DIR, "smallnorb", "smallnorb-{}-{}.mat")
SMALLNORB_CHUNKS = ["5x46789x9x18x6x2x96x96-training", "5x01235x9x18x6x2x96x96-testing"]


class SmallNORB(GroundTruthData):
    """SmallNORB dataset.
    The data set can be downloaded from
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. Images are resized to 64x64.
    The ground-truth factors of variation are:
    0 - category (5 different values)
    1 - elevation (9 different values)
    2 - azimuth (18 different values)
    3 - lighting condition (6 different values)
    The instance in each category is randomly sampled when generating the images.
    """

    def __init__(self, transforms):
        self.images, features = _load_small_norb_chunks(
            SMALLNORB_TEMPLATE, SMALLNORB_CHUNKS
        )
        self.transforms = transforms
        self.factor_sizes = [5, 10, 9, 18, 6]
        # Instances are not part of the latent space.
        self.latent_factor_indices = [0, 1, 2, 3, 4]
        self.num_total_factors = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes)

    @property
    def factors_num_values(self):
        return [self.factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return [64, 64, 1]

    def sample_factors(self, num, random_state=None):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num)

    def sample_observations_from_factors(
        self, factors, random_state=None, draw_label=False
    ):
        # all_factors = self.state_space.sample_all_factors(factors, random_state)
        inds = self.index.features_to_index(factors)
        n = len(factors)
        x = torch.empty(n, 1, 64, 64)
        for i in range(n):
            x[i] = self.transforms(
                np.reshape(self.images[inds[i]].astype(np.float32), (64, 64, 1))
            )
        out_dict = {"x": x}
        if draw_label:
            y = torch.empty(n, 4)
            for i in range(n):
                y[i] = torch.from_numpy(self.label_from_factors(factors[i]))
            out_dict["y"] = y
        return out_dict

    @staticmethod
    def label_from_factors(factors):
        object_type, background = (
            factors[1],
            factors[4],
        )  # 0 to 9 included, 0 to 5 included

        # factors should be flat
        y = np.empty(shape=(4,), dtype=np.int_)
        # label 1: vehicle + light background
        y[0] = 1 if object_type >= 5 and background >= 3 else 0
        # label 2: vehicle  + dark background
        y[1] = 1 if object_type >= 5 and background < 3 else 0
        # label 3: human/animal + light background
        y[2] = 1 if object_type < 5 or background >= 3 else 0
        # label 4: human/animal+dark background
        y[3] = 1 if background < 3 else 0
        return y


def _load_small_norb_chunks(path_template, chunk_names):
    """Loads several chunks of the small norb data set for final use."""
    list_of_images, list_of_features = _load_chunks(path_template, chunk_names)
    features = np.concatenate(list_of_features, axis=0)
    features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
    return np.concatenate(list_of_images, axis=0), features


def _load_chunks(path_template, chunk_names):
    """Loads several chunks of the small norb data set into lists."""
    list_of_images = []
    list_of_features = []
    for chunk_name in chunk_names:
        norb = _read_binary_matrix(path_template.format(chunk_name, "dat"))
        list_of_images.append(_resize_images(norb[:, 0]))
        norb_class = _read_binary_matrix(path_template.format(chunk_name, "cat"))
        norb_info = _read_binary_matrix(path_template.format(chunk_name, "info"))
        list_of_features.append(np.column_stack((norb_class, norb_info)))
    return list_of_images, list_of_features


def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with tf.gfile.GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double",
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data


def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = PIL.Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), PIL.Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images / 255.0


class SmallNORBDataset(Dataset):
    def __init__(self, groundtruth, train, test_size=0.1, random_state=12345):
        self.train = train
        self.groundtruth = groundtruth
        self.transforms = groundtruth.transforms
        if train:
            self.ind, _ = train_test_split(
                range(len(groundtruth.images)),
                test_size=test_size,
                random_state=random_state,
            )
        else:
            _, self.ind = train_test_split(
                range(len(groundtruth.images)),
                test_size=test_size,
                random_state=random_state,
            )
        self.n = len(self.ind)
        # check whether index is in allowed set ( train / test set)
        self.check_ind = torch.zeros(len(self.groundtruth.images))
        self.check_ind[self.ind] = 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        while True:
            factors = self.groundtruth.sample_factors(1)
            ind = self.groundtruth.index.features_to_index(factors)
            factors = factors[0]
            # check if image in train / test set
            if self.check_ind[ind] == 1:
                x = self.groundtruth.images[ind].astype(np.float32)
                x = self.transforms(np.reshape(x, (64, 64, 1)))
                label = self.groundtruth.label_from_factors(factors)
                return x, label

    @property
    def factor_data(self):
        return self.groundtruth
