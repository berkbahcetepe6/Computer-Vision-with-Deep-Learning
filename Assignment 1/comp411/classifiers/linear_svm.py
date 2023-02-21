from distutils import core
import numpy as np
from random import shuffle
import builtins


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                #For every update on the loss, we can update gradients of the correct 
                #and non-correct classes due to backpropagation
                dW[:, y[i]] -= X[i] #correct gradient update
                dW[:, j] += X[i] #incorrect gradient update 

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= num_train #averaging gradient with respect ot number of samples
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW += 2*reg*W #add derivative of the regularization
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    
    
    return loss, dW

def huber_loss_naive(W, X, y, reg):
    """
    Modified Huber loss function, naive implementation (with loops).
    Delta in the original loss function definition is set as 1.
    Modified Huber loss is almost exactly the same with the "Hinge loss" that you have 
    implemented under the name svm_loss_naive. You can refer to the Wikipedia page:
    https://en.wikipedia.org/wiki/Huber_loss for a mathematical discription.
    Please see "Variant for classification" content.
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    ###############################################################################
    # TODO:                                                                       #
    # Complete the naive implementation of the Huber Loss, calculate the gradient #
    # of the loss function and store it dW. This should be really similar to      #
    # the svm loss naive implementation with subtle differences.                  #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score
            if margin >= 0:
                if margin < 1:
                    loss += margin * margin
                    dW[:, y[i]] -= X[i]*2*(margin)
                    dW[:, j] += X[i]*2*(margin)
                else:
                    loss += 4*margin
                    dW[:, y[i]] -= X[i]*(4)
                    dW[:, j] += X[i]*(4)
                

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), y][:, np.newaxis] #true label scores
    margins = np.maximum(0, scores - correct_class_scores + 1)

    margins[range(num_train), y] = 0 #do not add true score margins
    loss = np.sum(margins, where=margins>0) / num_train

    loss += reg * np.sum(W * W)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dw = (margins>0).astype(int)
    dw[range(num_train), y] -= dw.sum(axis=1)
    dW = (X.T).dot(dw) / num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def huber_loss_vectorized(W, X, y, reg):
    """
    Structured Huber loss function, vectorized implementation.

    Inputs and outputs are the same as huber_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured Huber loss, storing the  #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), y][:, np.newaxis] #true label scores
    margins = np.zeros(scores.shape)
    mask = (scores - correct_class_scores >= 0) & (scores - correct_class_scores < 1)
    sec_mask = scores - correct_class_scores >= 1
    margins[mask] = np.power(scores - correct_class_scores, 2)[mask]
    margins[sec_mask] = (4*(scores - correct_class_scores))[sec_mask]
    margins[range(num_train), y] = 0


    loss = np.sum(margins) / num_train

    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dw = mask.astype(int) * (scores - correct_class_scores)*2
    dw[sec_mask] = (4) * (np.logical_not(mask).astype(int))[sec_mask]
    dw[range(num_train), y] -= dw.sum(axis=1)
    dW = (X.T).dot(dw) / num_train
    dW += 2 * reg * W

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
