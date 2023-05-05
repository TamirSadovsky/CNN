from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


def create_train_validation_loaders(
        dataset: Dataset,
        validation_ratio: float,
        batch_size: int = 100,
        num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO: Create two DataLoader instances, dl_train and dl_valid.
    # They should together represent a train/validation split of the given
    # dataset. Make sure that:
    # 1. Validation set size is validation_ratio * total number of samples.
    # 2. No sample is in both datasets. You can select samples at random
    #   from the dataset.

    # ====== YOUR CODE: ======
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_idx = int(validation_ratio * len(dataset))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    dl_train = DataLoader(dataset,batch_size = batch_size, num_workers = num_workers, sampler = train_sampler)
    dl_valid = DataLoader(dataset,batch_size = batch_size, num_workers = num_workers, sampler = val_sampler)
    # ========================

    return dl_train, dl_valid
