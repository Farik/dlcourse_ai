import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, Param


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
        self.W = Param(0.001 * np.random.randn(n_input, hidden_layer_size))
        self.B = Param(0.001 * np.random.randn(1, hidden_layer_size))
        self.Wh = Param(0.001 * np.random.randn(hidden_layer_size, n_output))
        self.bh = Param(0.001 * np.random.randn(1, n_output))
        self.cache = {}
        self.X = None

    def forward_linear(self, X, param_code):
        self.cache['X'+param_code] = X
        return np.dot(X, self.params()['W'+param_code].value) + self.params()['b'+param_code].value


    def backward_linear(self, A, param_code):
        A_prev = np.dot(A, self.params()['W'+param_code].value.T)
        self.params()['W'+param_code].grad = np.dot(self.cache['X'+param_code].T, A)
        self.params()['b'+param_code].grad = np.sum(A, axis=0, keepdims=True)

        return A_prev

    def forward_l2(self, X):
        return self.reg*np.sum(X**2)/2

    def backward_l2(self, A):
        return self.reg*A

    def forward_relu(self, X, param_code):
        self.cache['X'+param_code+'relu'] = X
        return np.maximum(0, X)

    def backward_relu(self, A, param_code):
        return np.multiply(A, np.int64(self.cache['X'+param_code+'relu'] > 0))

    def apply_grad(self, param_code):
        self.params()['W'+param_code].value += self.params()['W'+param_code].grad

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        for param_key in params:
            params[param_key].grad = np.zeros_like(params[param_key].value)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        X = self.X
        for layer in ['', 'h']:
            X = self.forward_linear(X, layer)
            X = self.forward_l2(X)
            X = self.forward_relu(X, layer)

        for layer in ['', 'h']:
            self.apply_grad(param_code=layer)

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

        raise Exception("Not implemented!")
        return pred

    def params(self):
        return {'W': self.W, 'B': self.B, 'Wh': self.Wh, 'bh': self.bh}
