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

    return reg_strength*np.sum(W**2)/2, reg_strength*W


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''

    exps = np.atleast_2d(np.exp(predictions - np.max(predictions)))
    return exps/np.sum(exps, axis=1)[:, None]


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    if not isinstance(target_index, np.ndarray):
        target_index = np.atleast_1d(target_index)
    probs = np.atleast_2d(probs.copy())
    batches_num = target_index.shape[0]

    log_likelihood = -np.log(probs[range(batches_num), target_index.squeeze()])
    return np.sum(log_likelihood)/batches_num

    #return -np.sum(target_index*np.log(probs))

    # m = 1 if not isinstance(target_index, np.ndarray) else target_index.shape[0]
    #
    # log_likelihood = -np.log(probs[range(m), target_index])
    # loss = np.sum(log_likelihood) / m
    # return loss


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

    if not isinstance(target_index, np.ndarray):
        target_index = np.atleast_2d(target_index)

    batches_num = target_index.shape[0]

    probs = softmax(predictions.copy())

    #Find indicies that you need to replace
    #inds = np.where(np.isnan(probs))

    #Place column means in the indices. Align the arrays using take
    #probs[inds] = 1/probs.shape[0]

    grad = np.atleast_2d(probs.copy())
    grad[range(batches_num), target_index.squeeze()] -= 1
    grad = grad/batches_num

    return cross_entropy_loss(probs, target_index), grad.reshape(predictions.shape)

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
        self.activation_cache = None
        pass

    def forward(self, X):
        self.activation_cache = X.copy()
        return np.maximum(0, self.activation_cache)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = np.multiply(d_out, np.int64(self.activation_cache > 0))
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.01 * np.random.randn(n_input, n_output))
        self.B = Param(0.01 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
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
        self.max_index = np.zeros((hp, wp),dtype=object)
        for y in range(0, height, s):
            for x in range(0, width, s):
                ys = y//self.stride
                xs = x//self.stride
                pool_frame_flat = self.X[:, y:y+p, x:x+p, :].transpose(0, 3, 1, 2).reshape(batch_size*channels, p**2)
                out[:, ys, xs, :] = np.amax(pool_frame_flat, axis=1).reshape(batch_size, channels)
                self.max_index[ys, xs] = np.where(pool_frame_flat == np.array(np.amax(pool_frame_flat, axis=1))[:, None])
                #self.max_index[ys, xs] = pool_frame_flat == np.array(np.amax(pool_frame_flat, axis=1))

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
                pool_max_index = self.max_index[ys, xs]
                pool_frame_flat = np.zeros((batch_size*channels, p**2))#d_input[:, y:y+p, x:x+p, :].transpose(0, 3, 1, 2).reshape(batch_size*channels, p**2)

                equal_max_counts = np.unique(self.max_index[ys, xs][:][0], return_counts=True)[1]
                pool_frame_flat[pool_max_index] = (d_out[:, ys, xs, :].reshape(batch_size*channels)/equal_max_counts)[pool_max_index[:][0]]

                d_input[:, y:y+p, x:x+p, :] += pool_frame_flat.reshape(batch_size, channels, p, p).transpose(0, 2, 3, 1)

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]

        return X.copy().reshape((batch_size, height*width*channels))

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
