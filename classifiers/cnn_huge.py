


import numpy as np


class ResNet:
    """
    Filter size: Above we used 7x7; this makes pretty pictures but smaller filters may be more efficient
    Number of filters: Above we used 32 filters. Do more or fewer do better?
    Batch normalization: Try adding spatial batch normalization after convolution layers and vanilla batch normalization aafter affine layers. Do your networks train faster?
    Network architecture: The network above has two layers of trainable parameters. Can you do better with a deeper network? You can implement alternative architectures in the file cs231n/classifiers/convnet.py. Some good architectures to try include:
        [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
        [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
        [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]"""

    def __init__(self,input_dim=(3, 32, 32),num_filters=(16,32,64), filter_size=3,hidden_dim_CNN=50,layers_base=3,
               hidden_dim_FC=100, num_classes=10, weight_scale=1e-3, reg=0.0,dropout=0, use_batchnorm=True,stride=2,feature_map_size={32,16,8},
               dtype=np.float32):
        """
            Initialize a new network.
            number of layers is     1 + ( 6 * layers_base ) + 1

            for each filter map size {32,16,8}
                (2 * layers_base) number of layers + 2


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
        self.params['stride'] = stride
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0

        C, H, W = input_dim

        N = layers_base * 6




        T = 0
        k = 0
        #i = 0
        self.params['W' + (str(1))] = np.random.normal(scale=weight_scale, size=(num_filters[1], input_dim[0], filter_size, filter_size))
        self.params['b' + (str(1))] = np.zeros(num_filters[1])
        for n in range(N):
            if n % (2 * layers_base) == 0 and n > 1:
                k += 1
            if n % (2 * layers_base) == 0 and n > 1:
                i = k - 1
                print i
            else:
                i = k
            self.params['W'+(str(n+2))] = np.random.normal(scale=weight_scale,size=(num_filters[k],num_filters[1],filter_size,filter_size))
            self.params['b'+(str(n+2))] = np.zeros(num_filters[1 ])
            if self.use_batchnorm:
                #self.params['gamma' + (str(n + 2))] = np.ones(hidden_dim_CNN[n])
                #self.params['Beta' + (str(n + 2))] = np.zeros(hidden_dim_CNN[n])
                pass
        self.params['W' + (str(N+2))] = np.random.normal(scale=weight_scale, size=(num_filters[i]*filter_size*filter_size,num_classes))

        self.params['b' + (str(N+2))] = np.zeros((num_classes))

        pass


    def loss(self,X,y=None):

        import Layers as layers
        from faster.fast_layers import conv_forward_fast
        ly = layers.Layers()
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        conv1 , chache = conv_forward_fast(X, self.params['W1'], self.params['b1'], conv_param)

        print "conv1"
        print conv1.shape

        print "w2"
        print self.params['W2'].shape

        out, cache1, rcache, cache2, rcahce2, x_pad  = ly.resNetUnit_forward(conv1, self.params['W2'] , self.params['b2'] ,self.params['W3'] , self.params['b3'] , conv_param )




        print conv1.shape





        pass
