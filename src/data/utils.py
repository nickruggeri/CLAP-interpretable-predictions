import random
from pathlib import Path

import numpy as np
from torch.utils.data.sampler import Sampler

DATA_DIR = Path("./data").resolve()


class SubsetSampler(Sampler):
    def __init__(self, num_samples, n):
        self.num_samples = num_samples
        self.list_n = np.arange(n)

    def __iter__(self):
        random.shuffle(self.list_n)
        return iter(self.list_n[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


# The following are utilities for datasets where the ground truth factors are known.
# https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/ground_truth_data.py#L22
class GroundTruthData(object):
    """Abstract class for data sets that are two-step generative models."""

    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factor_bases(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state=None):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(
        self, factors, random_state=None, draw_label=False
    ):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state=None, draw_label=False):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(
            factors, random_state, draw_label=draw_label
        )

    def sample_observations(self, num, random_state=None, draw_label=False):
        """Sample a batch of observations X."""
        return self.sample(num, random_state, draw_label=draw_label)[1]

    def idx_to_factors(self, idx):
        factors = np.zeros(shape=(self.num_factors,))
        for pos, factor_base in enumerate(self.factor_bases):
            factor = np.floor_divide(idx, factor_base)
            factors[pos] = factor
            idx -= factor * factor_base
        assert idx == 0, str(idx) + " remainder is not 0"
        return factors


class SplitDiscreteStateSpace:
    def __init__(self, factor_sizes):
        self.factor_sizes = factor_sizes
        self.num_factors = len(factor_sizes)

    def sample_latent_factors(self, num):
        """Sample a batch of the latent factors."""
        factors = np.zeros(shape=(num, self.num_factors), dtype=np.int64)
        for i in range(self.num_factors):
            factors[:, i] = self._sample_factor(i, num)
        return factors

    def _sample_factor(self, i, num):
        return np.random.randint(
            low=0, high=self.factor_sizes[i], size=(num,)
        )  # high is exclusive


class StateSpaceAtomIndex(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        """Creates the StateSpaceAtomIndex.
        Args:
          factor_sizes: List of integers with the number of distinct values for each
            of the factors.
          features: Numpy matrix where each row contains a different factor
            configuration. The matrix needs to cover the whole state space.
        """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned.
        """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factor configurations.
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the atom space should be
            returned.
        """
        if np.any(features > np.expand_dims(self.factor_sizes, 0)) or np.any(
            features < 0
        ):
            raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)
