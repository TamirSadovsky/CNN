from typing import Sequence, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import helpers.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor) -> Tensor:
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            k_nearest_neighbors, indices = torch.topk(dist_matrix[:, i], k=self.k, largest=False)
            k_nearest_labels = self.y_train[indices]
            y_pred[i] = torch.mode(k_nearest_labels).values

            # ========================
        return y_pred

    def calc_distances(self, x_test: Tensor) -> Tensor:
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        # Hint 1: Open the expression (a-b)^2.
        # Hint 2: Use "Broadcasting Semantics".

        dists = torch.tensor([])
        # ====== YOUR CODE: ======
        norms_train = torch.sum(self.x_train**2, dim = 1).unsqueeze(1)
        norms_test = torch.sum(x_test**2, dim = 1)
        multiply = self.x_train @ x_test.transpose(0,1)
        dists = torch.sqrt(norms_train - (2 * multiply) + norms_test.T)
        
        # ========================

        return dists


def accuracy(y: Tensor, y_pred: Tensor) -> float:
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction. (between 0 and 1)
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    correct_predictions = torch.sum(y == y_pred)
    accuracy = correct_predictions.item() / y.shape[0]
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices: Sequence[int], num_folds: int) -> Tuple[int, List[List[float]]]:
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    total_size = len(ds_train)
    fraction = 1 / num_folds
    segment_size = int(total_size * fraction)

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        accuracies_per_fold = []
        for j in range(num_folds):
            train_indices = np.delete(np.arange(total_size), np.arange(j * segment_size, (j+1) * segment_size))
            validation_indices = np.arange(j * segment_size, (j+1) * segment_size)

            train_data = torch.utils.data.dataset.Subset(ds_train, train_indices)
            validation_data = torch.utils.data.dataset.Subset(ds_train, validation_indices)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=2)
            validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=50, shuffle=True, num_workers=2)

            model.train(train_loader)
            # This converts the validation data from DataLoader format to two arrays (x_val and y_val) that the model can                 use for prediction.
            x_val, y_val = dataloader_utils.flatten(validation_loader)
            # This uses the trained model to predict the labels for the validation data.
            y_pred = model.predict(x_val)

            validation_accuracy = accuracy(y_val, y_pred)
            accuracies_per_fold.append(validation_accuracy)

        accuracies.append(accuracies_per_fold)

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
