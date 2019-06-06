import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.L = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
#            FullyConnectedLayer(n_input, n_output),
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
            forward_input = layer.forward(forward_input)

        loss, backward_propagation = softmax_with_cross_entropy(forward_input, y)

        for layer in reversed(self.L):
            backward_propagation = layer.backward(backward_propagation)

            for reg_param in ['W', 'B']:
                if reg_param in layer.params():
                    loss_l2, dp_l2 = l2_regularization(layer.params()[reg_param].value, self.reg)
                    loss += loss_l2
                    layer.params()[reg_param].grad += dp_l2

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
