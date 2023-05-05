from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier:
    def __init__(self, n_features: int, n_classes: int, weight_std: float = 0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weight_std = weight_std
        self.weights = torch.normal(mean=0, std=weight_std, size=(n_features, n_classes))
        # ========================

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        # calculates class scores by matrix multipication, resulting in a tensor of shape (N, n_classes), where N is the             batch size and n_classes is the number of classes.
        class_scores = torch.matmul(x, self.weights)
        # predicted class for each sample is then determined by taking the class with the highest score
        y_pred = torch.argmax(class_scores , dim=1)
        return y_pred, class_scores
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor) -> float:
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        correct_predictions = torch.sum(y == y_pred)
        acc = correct_predictions.item() / y.shape[0]
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        # Set model to train mode
        #self.train()

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            total_correct = 0
            average_loss = 0

            for xb, yb in dl_train:
                y_pred, class_scores = self.predict(xb)
                loss = loss_fn.loss(xb, yb, class_scores, y_pred) + weight_decay * torch.norm(self.weights)
                average_loss += loss
                total_correct += self.evaluate_accuracy(yb, y_pred)
                grad = loss_fn.grad()
                self.weights -= learn_rate * grad

            train_res.loss.append(average_loss / len(dl_train))
            train_res.accuracy.append(total_correct / len(dl_train))

            valid_accuracy = 0
            valid_loss_acc = 0
            for valid_x, valid_y in dl_valid:
                valid_y_pred, valid_class_scores = self.predict(valid_x)
                valid_accuracy += self.evaluate_accuracy(valid_y, valid_y_pred)
                valid_loss_acc += loss_fn.loss(valid_x, valid_y, valid_class_scores, valid_y_pred)

            valid_res.loss.append(valid_loss_acc / len(dl_valid))
            valid_res.accuracy.append(valid_accuracy / len(dl_valid))

            print('.', end='')
        print('')

        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        # Extract the weights tensor from the model
        # Reshape the weights tensor to be of shape (n_classes, C, H, W) by adding a new dimension at the beginning of the           tensor
        weights_tensor = self.weights[:-1] if has_bias else self.weights
        w_images = weights_tensor.T.reshape(self.n_classes, *img_shape)
        # ========================

        return w_images
