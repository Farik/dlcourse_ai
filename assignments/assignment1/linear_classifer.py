import numpy as np


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
    grad = np.atleast_2d(probs.copy())
    grad[range(batches_num), target_index.squeeze()] -= 1
    grad = grad/batches_num

    return cross_entropy_loss(probs, target_index), grad.reshape(predictions.shape)
    #return cross_entropy_loss(probs, target_index), np.sum(grad, axis=0)



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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''

    N, D = X.shape

    predictions = np.dot(X, W)

    if not isinstance(target_index, np.ndarray):
        target_index = np.atleast_2d(target_index)

    batches_num = target_index.shape[0]

    probs = softmax(predictions.copy())
    grad = np.zeros(W.shape)
    # grad[range(batches_num), target_index.squeeze()] -= 1
    # grad = np.dot(grad, X)/batches_num
    y = np.zeros(predictions.shape)
    y[range(target_index.shape[0]), target_index] = 1
    #grad = (-1 / N)*np.dot((y - probs), X)
    grad = (-1 / N) * np.dot(np.transpose(X),(y - probs))

    return cross_entropy_loss(probs, target_index), grad


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
