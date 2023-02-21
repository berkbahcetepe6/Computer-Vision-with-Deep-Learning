import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    C = np.max(y) + 1 #gets number of classes
    N, D = X.shape  # gets number of data intances and dimensions
    L_i = []
    
    for i in range(N):
        f = X[i].dot(W)
        f -= np.max(f) #to avoid numerical stability
        softmax = np.exp(f) / np.sum(np.exp(f))
        for c in range(C):
            if y[i] == c:
                L_i.append(-np.log(softmax[c]))
                loss += np.sum(L_i[i]) / N
                softmax[c] -= 1
        dW += np.outer(X[i], softmax) / N
    
    if regtype == 'L2':
        loss +=  reg_l2 * np.sum(W * W)
        dW += 2 * reg_l2 * W
    else:
        loss += reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))
        dW += 2 * reg_l2 * W +reg_l1 * np.sign(W)
                
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    C = np.max(y) + 1  # gets number of classes
    N, D = X.shape  # gets number of data intances and dimensions

    f = X.dot(W)
    f -= np.max(f, axis = 1, keepdims = True)
    softmax = np.exp(f) / np.sum(np.exp(f), axis = 1, keepdims= True)
    L_i = -np.log(softmax[range(N), y])
    loss += np.sum(L_i) / N

    softmax[range(N), y] -= 1
    dW = (X.T.dot(softmax)) / N

    if regtype == 'L2':
        loss +=  reg_l2 * np.sum(W * W)
        dW += 2 * reg_l2 * W
    else:
        loss += reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))
        dW += 2 * reg_l2 * W + reg_l1 * (np.sign(W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
