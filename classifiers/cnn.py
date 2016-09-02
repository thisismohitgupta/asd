import numpy as np


class ThreeLayerConvNet(object):
    """
     A three-layer convolutional network with the following architecture:

     conv - relu - 2x2 max pool - affine - relu - affine - softmax

     The network operates on minibatches of data that have shape (N, C, H, W)
     consisting of N images, each with height H and width W and with C input
     channels.
     """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,hidden_dim=100,
                 num_classes=10, weight_scale=1e-3,
                 reg=0.0,dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        W2_row_size = num_filters * input_dim[1] / 2 * input_dim[2] / 2
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(W2_row_size, hidden_dim))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax

        from Layers import Layers
        from faster.fast_layers import conv_forward_strides , conv_backward_strides , max_pool_forward_fast , max_pool_backward_fast



        ly = Layers()
        conv_l, Cconv_l = conv_forward_strides(X, W1, b1, conv_param)
        relu_l, Crelu_l = ly.relu_forward(conv_l)
        maxPool, Cmaxpool = max_pool_forward_fast(relu_l, pool_param)
        af_f, daf_f = ly.affine_forward(maxPool, W2, b2)
        relu_2, Crelu_2 = ly.relu_forward(af_f)
        scores, Cscores = ly.affine_forward(relu_2, W3, b3)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        if y is None:
            return scores
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, grad = ly.softmax_loss(scores, y)
        loss += sum(0.5 * self.reg * np.sum(W_tmp ** 2) for W_tmp in [W1, W2, W3])

        daffineB, dW3, db3 = ly.affine_backward(grad, Cscores)
        drelu_2 = ly.relu_backward(daffineB, Crelu_2)
        daffineC, dW2, db2 = ly.affine_backward(drelu_2, daf_f)
        dmax = max_pool_backward_fast(daffineC, Cmaxpool)
        drelu_l = ly.relu_backward(dmax, Crelu_l)
        dx, dw1, db1 = conv_backward_strides(drelu_l, Cconv_l)

        grads["W1"] = dw1
        grads["W2"] = dW2
        grads["W3"] = dW3
        grads["b1"] = db1

        grads["b2"] = db2
        grads["b3"] = db3

        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']

        grads['W1'] += self.reg * self.params['W1']



        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    pass