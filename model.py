#! /usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""
Contains neural network models for signal classification implemented
in the PyTorch framework.

When getting started with this package, you can use these models as-is:
the default sizes and parameter values are reasonable for the bundled
datasets. Next, you can explore how tuning the values provided as
arguments to these models ("model hyperparameters") affects their
performance and accuracy. Finally, these models are each standalone
functions, so you can copy them into your own code to add layers, change
activation functions, and experiment with different model structures
entirely!
"""

from typing import Optional, Tuple
import torch
import numpy as np
from torchinfo import summary

class default_network(torch.nn.Module):
    """ Creates a convolutional signal classifier model for chunks of signal
    data that are a specified number of samples long.

    Required arguments are the input length (number of samples in the input)
    and the output length number of classes of signal in the classifier. Other
    optional arguments are tunable hyperparameters to optimize the model
    for specific datasets and/or types of signals. When the optional arguments
    are set to ``None``, default values will be used that will work reasonably
    well.

    This model object is created with default-initialized weights and must be
    first trained before being used for inference.
    """
    def __init__(self,
                 input_len: int,
                 output_len: int,
                 num_filt: Optional[float] = 32,  # 필요에 따라 늘릴 수 있음
                 strides: Optional[tuple] = (32, 1),
                 filt_size: Optional[tuple] = (32, 2),
                 pool_size: Optional[tuple] = (16, 1),
                 fc_num_nodes: Optional[list] = [256,128, 64],
                 padding: Optional[tuple] = (0, 0)
                 ) -> None:
        """
        :param input_len:    Number of complex samples to be input to the model
        :param output_len:   Number of output classes for this classifier
        :param num_filt:     Number of output filters in the convolution
        :param strides:      Stride within convolutional layers
        :param filt_size:    Kernel/window size of convolutional layers,
                             typically ``(X, 2)`` for some ``X``
        :param pool_size:    Kernel/window size of downsampling layers,
                             typically ``(X, 1)`` for some ``X``
        :param fc_num_nodes: List of sizes of fully-connected (FC) layers
        :param padding:      Padding dimensions
        """
        super(default_network, self).__init__()
        self.input_len = input_len

        # Define the Feature Creation Network
        conv0 = torch.nn.Conv2d(1, num_filt, filt_size, strides, bias=True)
        relu0 = torch.nn.ReLU(inplace=True)
        pool0 = torch.nn.MaxPool2d(pool_size)
        flat0 = torch.nn.Flatten()
        self.fv = torch.nn.Sequential(conv0, relu0, pool0, flat0)

        # To determine the output size of the feature vector, the size of each layer
        # must be calculated sequentially. First, calculate output dimensions of the Conv2d layer
        c0, h0, w0 = calc_output_dims(self.input_len, 2, num_filt, filt_size, strides, padding)
        # Calculate output dimensions of the MaxPool2d layer
        c1, h1, w1 = calc_output_dims(h0, w0, c0, pool_size)
        # Calculate output len of the Flatten layer
        num_features = c1 * h1 * w1

        # Define the Classifier Network
        fc1 = torch.nn.Linear(num_features, fc_num_nodes[0], bias=True)
        relu1 = torch.nn.ReLU()
        fc2 = torch.nn.Linear(fc_num_nodes[0], fc_num_nodes[1], bias=True)
        relu2 = torch.nn.ReLU()
        fc3 = torch.nn.Linear(fc_num_nodes[1], fc_num_nodes[2], bias=True)
        relu3 = torch.nn.ReLU()
        # Define output layer
        fc4 = torch.nn.Linear(in_features=fc_num_nodes[2], out_features=output_len, bias=True)
        self.classifier = torch.nn.Sequential(fc1, relu1, fc2, relu2, fc3, relu3, fc4)

    def forward(self, x_real: torch.float32) -> torch.Tensor:
        """ Feed forward of the neural network

        .. note ::
            Data needs to be converted from complex tensor prior to this

        :param x_real:  non complex tensor
        :return:        output
        """
        x_reshape = x_real.view(-1, 1, self.input_len, 2)
        feature_vector = self.fv(x_reshape)
        output = self.classifier(feature_vector)
        return output

    def get_dummy_data(self, batch_size: int) -> torch.Tensor:
        """Get a tensor (containing random data) of the shape and data type expected
        by this model. This dummy data tensor is used to print a summary of the model as well as
        during tracing for ONNX model export and in test cases.

        .. note ::
            You can always test that the model is sized correctly to pass data by running
            ``model.forward(model.get_dummy_data(batch_size))``

        :param batch_size:  batch size
        :return:            input_data
        """
        buff = np.random.randn(batch_size, self.input_len * 2).astype(np.float32)
        input_data = torch.tensor(buff, dtype=torch.float32)
        return input_data

def calc_output_dims(h_in: int, w_in: int, c: int,
                     kernel_size: tuple,
                     stride: Optional[tuple] = None,
                     padding: Optional[tuple] = None,
                     dilation: Optional[tuple] = None) -> Tuple[int, int, int]:
    """ Calculate torch.nn.Conv2d or torch.nn.MaxPool2d output dimensions

    .. note ::
        Ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    :param h_in:        height dimensions of layer input
    :param w_in:        width dimension of layer input
    :param c:           channel dimension of layer output
    :param kernel_size: kernel dimensions
    :param stride:      stride dimensions
    :param padding:     padding dimensions
    :param dilation:    dilation dimensions
    :return:            output dimensions of layer (C, H, W)
    """
    # Get default values
    if stride is None:
        stride = kernel_size
    if padding is None:
        padding = np.zeros_like(kernel_size).tolist()
    if dilation is None:
        dilation = np.ones_like(kernel_size).tolist()
    # Calculate output dimensions
    c_out = int(c)
    h_out = int(((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    w_out = int(((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return c_out, h_out, w_out

if __name__ == '__main__':
    MODEL = default_network(input_len=1200000, output_len=20)
    MODEL.forward(MODEL.get_dummy_data(64))
    summary(model=MODEL, batch_size=4)
