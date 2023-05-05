import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        # TODO: Use Tensor.view() to implement the transform.
        # ====== YOUR CODE: ======
        # changes the view of the tensor to given view_dims view.
        return tensor.view(self.view_dims)
        # ========================


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Adds an element equal to 1 to
    a given tensor.
    """

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 1, "Only 1-d tensors supported"

        # TODO: Add a 1 at the end of the given tensor.
        # Make sure to use the same data type.

        # ====== YOUR CODE: ======
        # create new torch with the dimensions of the original tensor
        add_bias = torch.ones(1, dtype=tensor.dtype, device=tensor.device)
        # concatenates it with the input tensor along the first dimensions
        return torch.cat((tensor,add_bias))
        # ========================


