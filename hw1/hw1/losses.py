import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        num_samples = x_scores.shape[0]
        # generates a tensor of indices from 0 to num_samples - 1, which is used to index into the x_scores tensor along the         first dimension. y is used to index along the second dimension, which corresponds to the correct class for each             sample. The result is a tensor of shape (num_samples, 1).
        correct_class_scores = x_scores[torch.arange(num_samples), y].reshape(-1, 1)
        # the score for the correct class is subtracted from all scores, and the delta value is added. This generates a             matrix of margins, where each row corresponds to a sample, and each column corresponds to a class. 
        margins = x_scores - correct_class_scores + self.delta
        # any negative margins are set to 0. This ensures that the loss is 0 if the correct class score is already the               highest among all scores.
        margins[margins < 0] = 0

        # don't count the correct class in the loss
        margins[torch.arange(num_samples), y] = 0

        # calculate the average loss over the batch
        loss = torch.mean(torch.sum(margins, dim=1))

        # save necessary values for gradient calculation
        self.grad_ctx = {'x': x, 'y': y, 'x_scores': x_scores, 'y_predicted': y_predicted, 'margins': margins}
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        margins = self.grad_ctx['margins']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']

        num_samples = x.shape[0]

        # Compute binary matrix for each margin
        binary = margins
        binary[binary > 0] = 1
        binary[torch.arange(num_samples), y] = 0
        # set the values of the binary matrix to -row_sum at the positions corresponding to the correct classes. This               ensures that the gradients of the correct classes are proportional to the number of other classes that have a               positive margin with each sample.
        binary[torch.arange(num_samples), y] = -1 * torch.sum(binary, dim=1)

        # Compute the gradient
        grad = torch.mm(x.T, binary) / num_samples
        # ========================

        return grad
