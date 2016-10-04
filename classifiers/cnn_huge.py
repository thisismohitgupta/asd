


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
    
    _N , _num_classes ,  _hidden_dim_FC , _layers_base = 0 ,0 ,0 ,0

    
    def __init__(self,input_dim=(3, 32, 32),num_filters=(8,16,32), filter_size=3,hidden_dim_CNN=50,layers_base=3,
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
        #self.op_params['stride'] = stride
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0

        C, H, W = input_dim

        N = layers_base * 6
        
        self._N , self._num_classes ,  self._hidden_dim_FC , self._layers_base = N , num_classes , hidden_dim_FC , layers_base 



        T = 0
        k = 0
        #i = 0
        self.params['W' + (str(1))] = np.random.normal(scale=weight_scale, size=(num_filters[0], C, filter_size, filter_size))
        self.params['b' + (str(1))] = np.zeros(num_filters[0])
        
        
        for n in range(N):
            if n % (2 * layers_base) == 0 and n > 1:
                k += 1
                i = k - 1                
            else:
                i = k
            f = filter_size    
            self.params['W'+(str(n+2))] = np.random.normal(scale=weight_scale,size=(num_filters[k],num_filters[i],f,f))
            
            self.params['b'+(str(n+2))] = np.zeros(num_filters[k])
            
            if self.use_batchnorm:
                #self.params['gamma' + (str(n + 2))] = np.ones(hidden_dim_CNN[n])
                #self.params['Beta' + (str(n + 2))] = np.zeros(hidden_dim_CNN[n])
                pass
            
        print k    
        self.params['W' + (str(N+2))] = np.random.normal(scale=weight_scale, size=(num_filters[k] *num_filters[k-1]*(num_filters[k-2]*8),num_classes))

        self.params['b' + (str(N+2))] = np.zeros((num_classes))

        pass


    def loss(self,X,y=None):

        import Layers as layers
        from faster.fast_layers import conv_forward_fast, conv_backward_fast
        
        ly = layers.Layers()
        
        
        filter_size = self.params['W1'].shape[2]
        
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        
        
        
        ##################################################################################################################
        ##################################################################################################################
        ############################## Res net Forward computation          ##############################################
        ################################################################################################################
        ################################################################################################################
        
        ##first pass
        
        outputs = {}
        
        outputs['resNet_out_1'] , outputs['conv_cache_1'] = conv_forward_fast(X, self.params['W1'], self.params['b1'], conv_param)

        k = 0
        l = 1 
        
        #print self.params['b1']       
        for n in range(self._N / 2 ):
            
            if k < self._N  :
                i = str(k+2)
                j = str(k+3)
                m = str(l)
                n = str(l+1)
                o = str(k+1)
               
                #follow up pass
                #print 'W '+ i +' \n'
                
                outputs['resNet_out_'+ n], outputs['conv_layer_cache_' + o ], outputs['relu_cache_'+o], outputs['conv_layer_cache_'+i], outputs['relu_cache_'+i], outputs['x_pad_'+m]  = ly.resNetUnit_forward(
                    outputs['resNet_out_'+ m], 
                    self.params['W'+ i] , 
                    self.params['b'+ i] ,
                    self.params['W'+ j] , 
                    self.params['b'+ j] , 
                    conv_param )
            l += 1
            k += 2
               
        
        
        #print outputs['resNet_out_'+ str(10)].shape


        
        scores, scores_cache = ly.affine_forward(outputs['resNet_out_'+ str(l)] ,self.params['W'+ str(self._N+2)],self.params['b'+ str(self._N+2)])
        
        
        
        
        if y is None:
            return scores
        #print self.params['b1']
        
        loss,grad = ly.softmax_loss(scores,y)
        
        for W_tmp in self.params:
            if W_tmp[0] == 'W':
                loss += np.sum( 0.5 * self.reg * np.sum(np.power(self.params[W_tmp], 2)))
                
        ##################################################################################################################
        ##################################################################################################################
        ##############################   Resnet backwards for grads       ##############################################
        ################################################################################################################
        ################################################################################################################
        grads = {}
        backwards = {}
        name = str((self._N / 2) +1)
        backwards['resNet_Back_input_'+name], grads["W"+ str(self._N+2)], grads["b"+ str(self._N+2)] = ly.affine_backward(grad, scores_cache)
        
         
        i = 0 
        

        
        
        for a in range( self._N/2 , 0, -1):
            
            #print "hello " + str (a)
            ######to write this for loop
            
            
            
            f= str(a+1)
            i = str((a*2)-1)
            j = str(a*2)
            
            iGrad = "W"+ str((a*2))
            jGrad = "W"+ str((a*2)+1)
            iBGrad = "b"+ str((a*2))
            jBGrad = "b"+ str((a*2)+1)
            
            #print "meowww f=" + i + " ,j= "+ j
            backwards['resNet_Back_input_'+ str(a)] , grads[iGrad] , grads[iBGrad] , grads[jGrad] , grads[jBGrad] =  ly.resNetUnit_backward(               backwards['resNet_Back_input_'+f],                                                                                                                    outputs['conv_layer_cache_'+i],                                                                                                                      outputs['relu_cache_'+i],                                                                                                                        outputs['conv_layer_cache_'+j],                                                                                                                                                                                                                              outputs['relu_cache_'+j] )
                                                                                                                      
                                                                                                                      
                                                                                                                      
            
            
            
            pass
    
        
        ##################################################################################################################
        ##################################################################################################################
        ##############################  loss calulation for classifier     ##############################################
        ################################################################################################################
        ################################################################################################################
        
        dx,grads['W1'],grads['b1'] = conv_backward_fast ( backwards['resNet_Back_input_1'], outputs['conv_cache_1'])
        
        #print dx.shape , dw.shape,db.shape , X.shape
        
        #for i in grads:
        #    if grads[i].shape == self.params[i].shape:
        #        #print i + " is ok"
        #        #print grads[i].shape
        #    #else:
        #        #print i + " is not ok"
        #        #print grads[i].shape
        #        pass
        
        for W_tmp in self.params:
            if W_tmp[0] == 'W':
                grads[W_tmp] += self.reg*self.params[W_tmp]
        
        
        return loss, grads
        
        
        
        pass
