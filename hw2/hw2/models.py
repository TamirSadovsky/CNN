import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        for i, hidden_feature in enumerate(hidden_features):
            if i == 0:
                blocks.append(Linear(in_features, hidden_feature))
            else:
                blocks.append(Linear(hidden_features[i-1], hidden_feature))

            if activation == 'relu':
                blocks.append(ReLU())
            elif activation == 'sigmoid':
                blocks.append(Sigmoid())

            if dropout > 0:
                blocks.append(Dropout(dropout))

        blocks.append(Linear(hidden_features[-1], num_classes))
                    
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

        
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        self.h = in_h
        self.w = in_w
        
        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        filters = [in_channels] + self.filters

        for i, (in_channels, out_channels) in enumerate(zip(filters, filters[1:])):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) 
            layers.append(nn.ReLU())
            if i > 0 and (i + 1) % self.pool_every == 0 or i == len(filters) - 2:
                layers.append(nn.MaxPool2d(kernel_size=2))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        c, h, w = self.in_size
        conv2d_dilation, conv2d_padding, conv2d_stride = 1, 1, 1
        maxpool2d_kernel_size, maxpool2d_dilation, maxpool2d_padding = 2, 1, 0
        maxpool2d_stride = maxpool2d_kernel_size

        filters = [in_channels] + self.filters

        for i, (in_channels, out_channels) in enumerate(zip(filters, filters[1:])):
            kernel_size = 3
            h, w = [(x + 2 * conv2d_padding - conv2d_dilation * (kernel_size - 1) - 1) // conv2d_stride + 1 for x in (h, w)]
            c = out_channels

            if i > 0 and i % P == P - 1 or i == len(filters) - 2:
                h, w = [(x + 2 * maxpool2d_padding - maxpool2d_dilation * (maxpool2d_kernel_size - 1) - 1) // maxpool2d_stride + 1
                        for x in (h, w)]
                c = out_channels

        in_features = c * h * w

        hidden_dims = [in_features] + self.hidden_dims

        for in_feat, out_feat in zip(hidden_dims, hidden_dims[1:]):
            layers.append(nn.Linear(in_feat, out_feat))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        out = self.classifier(x)
    
        # ========================
        return out
    

class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        return self._make_layers(cfg['VGG13'])

    def _make_classifier(self):
        return nn.Linear(4608, 10) 

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv_layer = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                batchnorm_layer = nn.BatchNorm2d(x)
                activation_block = nn.ReLU(inplace=True)

                layers.extend([conv_layer, batchnorm_layer, activation_block])

                in_channels = x

        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)
    # ========================

