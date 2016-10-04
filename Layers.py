import numpy as np


class Layers:
    """ Layers for Computation"""

    def __init__(self):
        pass

    def affine_forward(self, x, w, b):
        """
        Computes the forward pass for an affine (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
        We multiply this against a weight matrix of shape (D, M) where
        D = \prod_i d_i
        Inputs:
        x - Input data, of shape (N, d_1, ..., d_k)
        w - Weights, of shape (D, M)
        b - Biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """

        cache = x, w, b

        out = x.reshape(x.shape[0], np.prod(x.shape[1:])).dot(w) + b

        return out, cache

    def affine_backward(self, dout, cache):
        """
        Computes the backward pass for an affine layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)
        - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """

        x, w, b = cache

        dx = dout.dot(w.T).reshape(x.shape)
        dw = x.reshape(x.shape[0], np.prod(x.shape[1:])).T.dot(dout)
        db = np.sum(dout,axis=0)

        return dx, dw, db

    def relu_forward(self, x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = np.maximum(0,x)
        cache = x
        return out,cache

    def relu_backward(self,dout,cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """

        dx = dout * (cache > 0)

        return dx


    def batchnorm_forward(self,x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance are
        computed from minibatch statistics and used to normalize the incoming data.
        During training we also keep an exponentially decaying running mean of the mean
        and variance of each feature, and these averages are used to normalize data
        at test-time.

        At each timestep we update the running averages for mean and variance using
        an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using a
        large number of training images rather than using a running average. For
        this implementation we have chosen to use running averages instead since
        they do not require an additional estimation step; the torch7 implementation
        of batch normalization also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """

        mode = bn_param['mode']

        eps = bn_param.get('eps', 1e-5)

        momentum = bn_param.get('momentum', 0.9)

        print x.shape
        N, D = x.shape

        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))

        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        out, cache = None, None
        if mode == 'train':
            #############################################################################
            # TODO: Implement the training-time forward pass for batch normalization.   #
            # Use minibatch statistics to compute the mean and variance, use these      #
            # statistics to normalize the incoming data, and scale and shift the        #
            # normalized data using gamma and beta.                                     #
            #                                                                           #
            # You should store the output in the variable out. Any intermediates that   #
            # you need for the backward pass should be stored in the cache variable.    #
            #                                                                           #
            # You should also use your computed sample mean and variance together with  #
            # the momentum variable to update the running mean and running variance,    #
            # storing your result in the running_mean and running_var variables.        #
            #############################################################################



            batch_mean = np.sum(x, axis=0) / N

            mewMean = x - batch_mean

            mewP2 = (mewMean) ** 2

            batch_varience = (np.sum(mewP2, axis=0) / N) + eps

            varience_sqroot = (np.sqrt(batch_varience ))

            oneDvarience_sqroot = 1.0 / varience_sqroot

            running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            running_var = momentum * running_var + (1 - momentum) * batch_varience

            norm = ((x - batch_mean) * oneDvarience_sqroot)

            out = (gamma * norm) + beta

            cache = norm, gamma, mewMean, oneDvarience_sqroot, varience_sqroot, batch_varience, eps, x
            pass
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
        elif mode == 'test':
            #############################################################################
            # TODO: Implement the test-time forward pass for batch normalization. Use   #
            # the running mean and variance to normalize the incoming data, then scale  #
            # and shift the normalized data using gamma and beta. Store the result in   #
            # the out variable.                                                         #
            #############################################################################
            norm = (x - running_mean) / (np.sqrt(running_var) + eps)

            out = (gamma * norm) + beta

            pass
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        return out, cache

    def batchnorm_backward(self,dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a computation graph for
        batch normalization on paper and propagate gradients backward through
        intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #############################################################################
        # TODO: Implement the backward pass for batch normalization. Store the      #
        # results in the dx, dgamma, and dbeta variables.                           #
        #############################################################################


        xhat, gamma, xmu, ivar, sqrtvar, var, eps, h = cache
        N, D = dout.shape
        dbeta = np.sum(dout, axis=0)
        dgammax = dout
        # step8
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        return dx, dgamma, dbeta

    def batchnorm_backward_alt(self,dout, cache):
        """
        Alternative backward pass for batch normalization.

        For this implementation you should work out the derivatives for the batch
        normalizaton backward pass on paper and simplify as much as possible. You
        should be able to derive a simple expression for the backward pass.

        Note: This implementation should expect to receive the same cache variable
        as batchnorm_backward, but might not use all of the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        #############################################################################
        # TODO: Implement the backward pass for batch normalization. Store the      #
        # results in the dx, dgamma, and dbeta variables.                           #
        #                                                                           #
        # After computing the gradient with respect to the centered inputs, you     #
        # should be able to compute gradients with respect to the inputs in a       #
        # single statement; our implementation fits on a single 80-character line.  #
        #############################################################################
        norm, gamma, mean, oneDivideVarience_sqroot, varience_sqroot, varience, eps, x = cache
        N, D = dout.shape

        dxi = dout * gamma

        df = dxi*( mean ) * ((-1.0 / 2.0) * (( varience )**(-3/2)))
        dvarience = np.sum(df, axis=0)

        dmew = (dxi * (- oneDivideVarience_sqroot ))
        dmew = np.sum(dmew , axis=0)

        dx = (dxi * oneDivideVarience_sqroot) + (dvarience * ((2.0*mean)/N)) + (dmew * (1.0 / N))
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * norm,axis=0)

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return dx, dgamma, dbeta

    def dropout_forward(self,x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We drop each neuron output with probability p.
          - mode: 'test' or 'train'. If the mode is train, then perform dropout;
            if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking but not in
            real networks.

        Outputs:
        - out: Array of the same shape as x.
        - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
          mask that was used to multiply the input; in test mode, mask is None.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            np.random.seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ###########################################################################
            # TODO: Implement the training phase forward pass for inverted dropout.   #
            # Store the dropout mask in the mask variable.                            #
            ###########################################################################

            mask = (np.random.rand(*x.shape) < p) / p
            out = x * mask

            pass
            ###########################################################################
            #                            END OF YOUR CODE                             #
            ###########################################################################
        elif mode == 'test':
            ###########################################################################
            # TODO: Implement the test phase forward pass for inverted dropout.       #
            ###########################################################################
            out = x

            pass
            ###########################################################################
            #                            END OF YOUR CODE                             #
            ###########################################################################

        cache = (dropout_param, mask)
        out = out.astype(x.dtype, copy=False)

        return out, cache

    def dropout_backward(self,dout, cache):
        """
        Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from dropout_forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################################
            # TODO: Implement the training phase backward pass for inverted dropout.  #
            ###########################################################################

            #dDivp = dout / dropout_param['p']
            print mask
            dx = mask * dout
            pass
            ###########################################################################
            #                            END OF YOUR CODE                             #
            ###########################################################################
        elif mode == 'test':
            dx = dout
        return dx

    def softmax_loss (self, x, y):

        prob = np.exp(x - np.max(x,axis=1,keepdims=True))
        prob /= np.sum(prob,axis=1,keepdims=True)
        N = x.shape[0]
        loss = -(np.sum(np.log(prob[np.arange(N),y]))) / N

        dx = prob
        dx[np.arange(N),y] -= 1
        dx /= N
        return loss, dx

    def conv_forward_naive( self, x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and width
        W. We convolve each input with F different filters, where each filter spans
        all C channels and has height HH and width HH.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """

        #############################################################################
        # TODO: Implement the convolutional forward pass.                           #
        # Hint: you can use the function np.pad for padding.                        #
        #############################################################################
        F, C, HH , WW = w.shape

        N, C, H, W = x.shape

        stride = conv_param["stride"]
        pad = conv_param["pad"]
        WandH = 1 + ((H + ( pad * 2 ) - HH) / stride)   #usually square images so no need to add useless computation
        z = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
        out = np.zeros((N,F,WandH,WandH))

        for n in range(N):

            z[n, :,:,:] = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
            for f in range(F):
                for hh in range(WandH):
                    for ww in range(WandH):


                        widthStart = stride * ww
                        widthEnd = WW + widthStart
                        heightStart = hh * stride
                        heightEnd = HH + heightStart

                        out[n,f,hh,ww] = np.sum(z[n, :, heightStart:heightEnd, widthStart:widthEnd] * (w[f, :])) + b[f]


        cache = x, w, b, conv_param
        return out, cache

    def conv_backward_naive( self, dout, cache):
        """
          A naive implementation of the backward pass for a convolutional layer.

          Inputs:
          - dout: Upstream derivatives.
          - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

          Returns a tuple of:
          - dx: Gradient with respect to x
          - dw: Gradient with respect to w
          - db: Gradient with respect to b
          """
        dx, dw, db = None, None, None
        #############################################################################
        # TODO: Implement the convolutional backward pass.                          #
        #############################################################################
        x, w, b ,conv_param = cache

        N, C, H, W = x.shape

        F, _, HH, WW = w.shape  # _  = C since the depth in a convolution layer remains the same

        _, _, H_prime, W_prime = dout.shape

        stride = conv_param['stride']
        pad = conv_param['pad']

        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        for n in range(N):
            dx_pad = np.pad(dx[n,:,:,:],((0,0),(pad,pad),(pad,pad)) ,"constant")
            x_pad = np.pad(x[n,:,:,:],((0,0),(pad,pad),(pad,pad)) ,"constant")

            for f in range(F):

                for h_prime in range(H_prime):

                    for w_prime in range(W_prime):
                        h1 = h_prime * stride
                        h2 = h_prime * stride + HH
                        w1 = w_prime * stride
                        w2 = w_prime * stride + WW

                        db[f] += dout[n,f,h_prime,w_prime]

                        dw[f,:,:,:] += x_pad[:,h1:h2,w1:w2] * dout[n,f,h_prime,w_prime]

                        dx_pad[:,h1:h2,w1:w2] += w[f,:,:,:] * dout[n,f,h_prime,w_prime]
            dx[n,:,:,:] = dx_pad[:,pad:-pad,pad:-pad]  # depad the output


        print dw.shape
        return dx, dw, db

    def max_pool_forward_naive(self,x, pool_param):
        """
        A naive implementation of the forward pass for a max pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
        """
        out = None

        #############################################################################
        # TODO: Implement the max pooling forward pass                              #
        #############################################################################
        N, C, H, W = x.shape

        p_h = pool_param['pool_height']
        p_w = pool_param['pool_width']
        stride = pool_param['stride']

        o_h = 1 + (H - p_h) / stride
        o_w = 1 + (W - p_w) / stride

        out = np.zeros((N, C, o_h, o_w))
        for n in range(N):
            for c in range(C):
                for h in range(o_h):
                    for w in range(o_w):
                        # out[n,:,h,w] = np.argmax(x[n,:, (h*stride):(( h * stride)) + p_h , (w*stride) : (w*stride) + p_w ])
                        window = x[n, :, (h * stride):((h * stride)) + p_h, (w * stride): (w * stride) + p_w]
                        out[n, :, h, w] = np.max(window.reshape((C, p_h * p_w)), axis=1)


        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = (x, pool_param)
        return out, cache


    def max_pool_backward_naive(self,dout, cache):
        """
        A naive implementation of the backward pass for a max pooling layer.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """

        #############################################################################
        # TODO: Implement the max pooling backward pass                             #
        #############################################################################
        x, pool_param = cache
        N, C, H, W = x.shape
        dx = np.zeros_like(x)
        p_h = pool_param['pool_height']
        p_w = pool_param['pool_width']
        stride = pool_param['stride']

        o_h = 1 + (H - p_h) / stride
        o_w = 1 + (W - p_w) / stride

        for n in range(N):
            for c in range(C):
                for h in range(o_h):
                    for w in range(o_w):
                        h1 = h * stride
                        h2 = h * stride + p_h
                        w1 = w * stride
                        w2 = w * stride + p_w
                        window = x[n, c, h1:h2, w1:w2]

                        window2 = np.reshape(window, p_h * p_w)

                        window3 = np.zeros_like(window2)
                        window3[np.argmax(window2)] = 1

                        dx[n, c, h1:h2, w1:w2] = np.reshape(window3, (p_h, p_w)) * dout[n, c, h, w]

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx

    def spatial_batchnorm_forward(self,x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0 means that
            old information is discarded completely at every time step, while
            momentum=1 means that new information is never incorporated. The
            default of momentum=0.9 should work well in most situations.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        #############################################################################
        # TODO: Implement the forward pass for spatial batch normalization.         #
        #                                                                           #
        # HINT: You can implement spatial batch normalization using the vanilla     #
        # version of batch normalization defined above. Your implementation should  #
        # be very short; ours is less than five lines.                              #
        #############################################################################
        N, C, H, W = x.shape
        temp, cache = self.batchnorm_forward(x.transpose(0, 3, 2, 1).reshape(N * W * H, C), gamma, beta, bn_param)

        out = temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return out, cache




    def spatial_batchnorm_backward(self,dout, cache):
        """
        Computes the backward pass for spatial batch normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #############################################################################
        # TODO: Implement the backward pass for spatial batch normalization.        #
        #                                                                           #
        # HINT: You can implement spatial batch normalization using the vanilla     #
        # version of batch normalization defined above. Your implementation should  #
        # be very short; ours is less than five lines.                              #
        #############################################################################
        # N,C,H,W = dout.shape
        # temp,dgamma,dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape(N*H*W,C),cache)

        # dx = temp.reshape(N,W,H,C).transpose(0,3,2,1)
        N, C, H, W = dout.shape
        dx_temp, dgamma, dbeta = self.batchnorm_backward_alt(dout.transpose(0, 3, 2, 1).reshape((N * W * H, C)), cache)
        dx = dx_temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return dx, dgamma, dbeta

    def resNetUnit_forward( self, x, w1, b1, w2, b2, conv_param):
        """
            A naive implementation of the backward pass for a convolutional layer.

            Inputs:
            - dout: Upstream derivatives.
            - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

            Returns a tuple of:
            - dx: Gradient with respect to x
            - dw: Gradient with respect to w
            - db: Gradient with respect to b
        """
        from faster.fast_layers import conv_forward_fast

        conv_layer_1 , conv_layer1_cache = conv_forward_fast(x, w1, b1, conv_param)

        relu , relu_cache = self.relu_forward(conv_layer_1)
             
       
        conv_layer_2, conv_layer2_cache = conv_forward_fast(relu, w2, b2, conv_param)

        #print conv_layer_1.shape

        if conv_layer_2.shape[1] > x.shape[1]:
            #print "shape is different"
            
            pad = (conv_layer_2.shape[1] - x.shape[1]) / 2
            x_pad = np.pad(x, ((0, 0), (pad ,pad ), (0, 0), (0, 0)), 'constant', constant_values=0)
            

        else:
            #print "shape is OK"
            x_pad = x

        

        addition = conv_layer_2 + x_pad

        out, relu_cache_2 = self.relu_forward(addition)

        return out, conv_layer1_cache, relu_cache, conv_layer2_cache, relu_cache_2, x_pad
    
    
    
    

    def resNetUnit_backward(self,dout, cache1, rcache, cache2, rcahce2):

        from faster.fast_layers import conv_backward_fast

        drelu2 = self.relu_backward(dout, rcahce2)

        dout = drelu2

        dconv2,dw2,db2 = conv_backward_fast(drelu2,cache2)


        drelu1 = self.relu_backward(dconv2, rcache)

        dx, dw1, db1 = conv_backward_fast(drelu1,cache1)


        if dout.shape[1] > dx.shape[1]:
            pad = dout.shape[1] - dx.shape[1]

            dout_unpad = dout[:,(dout.shape[1]-pad)/2:-((dout.shape[1]-pad)/2),:,:]
            #dout_unpad = dout
            
        else:
            dout_unpad = dout

        #print dout[:,0,:,:] 
        dout_unpad += dx

        return dout_unpad , dw1 , db1 , dw2 , db2
