import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    raise Exception("Not implemented!")
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        raise Exception("Not implemented!")        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        # batch_size, height, width, channels = X.shape
        batch_size, height, width, channels = X.shape
        self.X = np.pad(X, (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0)
        ), 'constant')

        out_height = self.X.shape[1] - self.filter_size + 1
        out_width = self.X.shape[2] - self.filter_size + 1

        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        i = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                # i[:, y, x] = np.sum(X[:, slice(y, y+self.filter_size), slice(x, x+self.filter_size), :]
                #                                     * self.W.value, axis=(1, 2)) + self.B.value
                x_i = np.reshape(self.X[:, slice(y, y+self.filter_size), slice(x, x+self.filter_size), :],
                                 (batch_size, self.filter_size*self.filter_size*self.in_channels)) #out_height*out_width*self.in_channels
                w_i = np.reshape(self.W.value, (self.filter_size*self.filter_size*self.in_channels, self.out_channels))

                i[:, y, x] = (np.dot(x_i, w_i) + self.B.value) #.reshape((batch_size, out_height, out_width, self.out_channels))
        return i

        # x_i = np.reshape(self.X, (batch_size, self.filter_size*self.filter_size*self.in_channels)) #out_height*out_width*self.in_channels
        # w_i = np.reshape(self.W.value, (self.filter_size*self.filter_size*self.in_channels, self.out_channels))

        # return (np.dot(x_i, w_i) + self.B.value).reshape((batch_size, out_height, out_width, self.out_channels))


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        d_input = np.zeros(self.X.shape)
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                x_i = np.reshape(self.X[:, slice(y, y+self.filter_size), slice(x, x+self.filter_size), :],
                                 (batch_size, self.filter_size*self.filter_size*self.in_channels))
                w_i = np.reshape(self.W.value, (self.filter_size*self.filter_size*self.in_channels, self.out_channels))

                d_out_i = d_out[:, y, x, :]

                d_x_i = np.dot(d_out_i.reshape(batch_size, self.out_channels), w_i.T)
                d_input[:, slice(y, y+self.filter_size), slice(x, x+self.filter_size), :] \
                    += np.reshape(d_x_i, (batch_size, self.filter_size, self.filter_size, self.in_channels))

                d_w_i = np.dot(x_i.T, d_out_i.reshape(batch_size, self.out_channels))
                self.W.grad += np.reshape(d_w_i, (self.filter_size, self.filter_size, self.in_channels, self.out_channels))

        self.B.grad = np.sum(d_out, axis=(0, 1, 2))

        if self.padding > 0:
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return d_input


        # x_i = np.reshape(self.X, (batch_size, self.filter_size*self.filter_size*self.in_channels))
        # w_i = np.reshape(self.W.value, (self.filter_size*self.filter_size*self.in_channels, self.out_channels))
        # d_out_i = np.reshape(d_out, (batch_size*out_height*out_width, self.out_channels))
        #
        # d_input = np.dot(d_out_i, w_i.T)
        # w_i_grad = np.dot(x_i.T, d_out_i)
        #
        # self.W.grad += w_i_grad.reshape(self.W.grad.shape)
        # self.B.grad += np.sum(d_out, axis=0, keepdims=True).reshape(self.B.grad.shape)
        #
        # return d_input.reshape(self.X.shape)

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()

        s = self.stride
        p = self.pool_size
        hp = height//p
        wp = width//p
        out = np.zeros((batch_size, hp, wp, channels))
        self.max_index = np.zeros((hp, wp, batch_size*channels), int)
        for y in range(0, height, s):
            for x in range(0, width, s):
                ys = y//self.stride
                xs = x//self.stride
                pool_frame = self.X[:, y:y+p, x:x+p, :].reshape(batch_size*channels, p**2)
                max_index = np.argmax(pool_frame, axis=1)
                out[:, ys, xs, :] = pool_frame[(range(pool_frame.shape[0]), max_index)].reshape(batch_size, channels)
                self.max_index[ys, xs] = max_index

        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        d_input = np.zeros(self.X.shape)

        s = self.stride
        p = self.pool_size
        for ys in range(d_out.shape[1]):
            for xs in range(d_out.shape[2]):
                y = ys*s
                x = xs*s
                max_index = self.max_index[ys, xs]
                pool_frame = d_input[:, y:y+p, x:x+p, :].reshape(batch_size*channels, p**2)
                pool_frame[(range(pool_frame.shape[0]), max_index)] = d_out.reshape(max_index.shape)
                d_input[:, y:y+p, x:x+p, :] += pool_frame.reshape(d_input.shape)

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
