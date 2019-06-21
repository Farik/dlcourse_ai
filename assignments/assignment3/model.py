import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """

        height,width,input_channels = input_shape
        self.L = [
            ConvolutionalLayer(in_channels=input_channels, out_channels=conv1_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(8, n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        for index, param in self.params().items():
            param.grad = np.zeros_like(param.value)

        forward_input = X.copy()
        for layer in self.L:
            #print(forward_input.shape)
            #print(layer)
            forward_input = layer.forward(forward_input)

        loss, backward_propagation = softmax_with_cross_entropy(forward_input, y)

        for layer in reversed(self.L):
            #print(backward_propagation.shape)
            #print(layer)
            backward_propagation = layer.backward(backward_propagation)

            # for reg_param in ['W', 'B']:
            #     if reg_param in layer.params():
            #         loss_l2, dp_l2 = l2_regularization(layer.params()[reg_param].value, self.reg)
            #         loss += loss_l2
            #         layer.params()[reg_param].grad += dp_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        forward_input = X.copy()
        for layer in self.L:
            forward_input = layer.forward(forward_input)

        pred = softmax(forward_input.copy())

        return np.argmax(pred, axis=1)

    def params(self):
        result = {}

        for layer_id, layer in enumerate(self.L):
            for key, value in layer.params().items():
                result[key+str(layer_id)] = value

        return result
