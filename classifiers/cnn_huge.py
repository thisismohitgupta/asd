


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

    
    def __init__(self,input_dim=(3, 32, 32),num_filters=(16,32,64), filter_size=3,hidden_dim_CNN=50,layers_base=3,
               hidden_dim_FC=100, num_classes=10, weight_scale=1e-3, reg=0.0,dropout=0, use_batchnorm=True,stride=2,feature_map_size=(32,16,8),mode='train',
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
        self.mode = mode
        
        C, H, W = input_dim

        N = layers_base * 6
        
        self._N , self._num_classes ,  self._hidden_dim_FC , self._layers_base = N , num_classes , hidden_dim_FC , layers_base 

        self.bn_params = []
        if self.use_batchnorm:
              self.bn_params = [{'mode': 'train'} for i in xrange(self._N +1)]
        
        #if self.use_batchnorm:
              #for bn_param in self.bn_params:
                    #bn_param[self.mode] = self.mode

        T = 0
        k = 0
        #i = 0
        self.params['W' + (str(1))] = np.random.normal(scale=weight_scale, size=(num_filters[0], C, filter_size, filter_size))
        self.params['b' + (str(1))] = np.zeros(num_filters[0])
        
        self.params['gamma' + str(0)] = np.ones(num_filters[0])
        self.params['Beta' + str(0)] = np.zeros(num_filters[0])
        
        for n in range(N):
            if n % (2 * layers_base) == 0 and n > 1:
                k += 1
                i = k - 1                
            else:
                i = k
            f = filter_size
            
            self.params['W'+(str(n+2))] = np.random.normal(scale=weight_scale,size=(num_filters[k],num_filters[i],f,f))
            #print n, self.params['W'+(str(n+2))].shape
            self.params['b'+(str(n+2))] = np.zeros(num_filters[k])
            
            if self.use_batchnorm:
                self.params['gamma' + (str(n + 1))] = np.ones(num_filters[k])
                self.params['Beta' + (str(n + 1))] = np.zeros(num_filters[k])
                pass
            
        #print k    
        self.params['W' + (str(N+2))] = np.random.normal(scale=weight_scale, size=(num_filters[2]*1*1,num_classes))
        #print self.params['W' + (str(N+2))]
        self.params['b' + (str(N+2))] = np.zeros((num_classes))
        
        self.params['CW0'] = np.random.normal(scale=weight_scale,size=(32,16,1,1))
            #print n, self.params['W'+(str(n+2))].shape
        self.params['Cb0'] = np.zeros(32)
        
        
        self.params['CW1'] = np.random.normal(scale=weight_scale,size=(32,16,1,1))
            #print n, self.params['W'+(str(n+2))].shape
        self.params['Cb1'] = np.zeros(32)
        
        
        self.params['CW2'] = np.random.normal(scale=weight_scale,size=(64,32,1,1))
            #print n, self.params['W'+(str(n+2))].shape
        self.params['Cb2'] = np.zeros(64)
        
        
        
        
        
        for k, v in self.params.iteritems():
              self.params[k] = v.astype(dtype)
        pass


    
    
    
    
    
    
    def loss(self,X,y=None):

        import Layers as layers
        from faster.fast_layers import conv_forward_fast, conv_backward_fast ,avg_pool_forward_fast,avg_pool_backward_fast
        mode = 'test' if y is None else 'train'
        ly = layers.Layers()
        #if self.use_batchnorm:
        #      self.bn_params = [{'mode': mode} for i in xrange(self._N +1)]
        
        filter_size = self.params['W1'].shape[2]
        
        conv_param = {'stride': 1, 'pad': 1}
        
        if self.use_batchnorm:
              for bn_param in self.bn_params:
                      bn_param['mode'] = mode
        
        ##################################################################################################################
        ##################################################################################################################
        ############################## Res net Forward computation          ##############################################
        ################################################################################################################
        ################################################################################################################
        
        ##first pass
        
        outputs = {}
        
        outputs['resNet_out_1_'] , outputs['conv_cache_1'] = conv_forward_fast(X, self.params['W1'], self.params['b1'], conv_param)
        
        
        
        outputs['resNet_out_1'],outputs['batch_catch'] = ly.spatial_batchnorm_forward(outputs['resNet_out_1_'],self.params['gamma' + str(0)],self.params['Beta' + str(0)],self.bn_params[self._N])
        
        
        k = 0
        l = 1 
        p = 0
        q = 0
        #print self.params['b1']       
        for s in range(self._N / 2 ):
            
            if k < self._N  :
                i = str(k+2)
                j = str(k+3)
                m = str(l)
                n = str(l+1)
                o = str(k+1)
               
                #follow up pass
                #print 'W '+ i +' \n'
                #print n %6
                if (s*2 % 6 == 0) and s *2 > 1 :
                    conv_param1 = {'stride': 2, 'pad': 0}
                    conv_param2 = {'stride': 1, 'pad': 1}
                    q += 1
                    p = q
                    
                else:
                    conv_param1 = {'stride': 1, 'pad': 1}
                    conv_param2 = {'stride': 1, 'pad': 1}
                    p = 0 
                
                f = str(p)
                #print "f is" + f 
                
                outputs['resNet_out_'+ n],                                          outputs['conv_layer_cache_' + o ],                                                outputs['relu_cache_'+o],                                                          outputs['conv_layer_cache_'+i],                                                           outputs['relu_cache_'+i],                                          outputs['x_pad_'+f],                                                                   outputs['bachnorm_layer_cache'+o] ,                                                              outputs['bachnorm_layer_cache'+i] = ly.resNetUnit_forward(
                    outputs['resNet_out_'+ m], 
                    self.params['W'+ i] , 
                    self.params['b'+ i] ,
                    self.params['W'+ j] , 
                    self.params['b'+ j] ,
                    self.params['gamma'+o],
                    self.params['Beta'+o],
                    self.params['gamma'+i],
                    self.params['Beta'+i],
                    self.bn_params[k],
                    self.bn_params[k+1],
                    self.params['CW'+f],
                    self.params['Cb'+f],
                    conv_param1,
                    conv_param2 )
                #print self.bn_params[k]['mode']
                #print 'bachnorm_layer_cache'+o , 'bachnorm_layer_cache'+i
            l += 1
            k += 2
               
        
        
        #print outputs['resNet_out_'+ str(10)].shape


        PoolParams = {'stride': 8, 'pool_width': 8, 'pool_height': 8}
        avg_pool_output,avg_pool_cache = avg_pool_forward_fast(outputs['resNet_out_'+ str(l)],pool_param=PoolParams)
        scores, scores_cache = ly.affine_forward( avg_pool_output,self.params['W'+ str(self._N+2)],self.params['b'+ str(self._N+2)])
        
        
        
        
        if y is None:
            return scores
        #print self.params['b1']
        
        loss,grad = ly.log_softmax_loss(scores,y)
        
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
        affine , grads["W"+ str(self._N+2)], grads["b"+ str(self._N+2)] = ly.affine_backward(grad, scores_cache)
        backwards['resNet_Back_input_' + name] = avg_pool_backward_fast(affine,avg_pool_cache)
         
        i = 0 
        
        k = 0
        
        l = 3 
        
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
            gammaGrad = "gamma"+ str(i)
            gammaGrad2 = "gamma"+ str(j)
            betaGrad = "Beta"+ str(i)
            betaGrad2 = "Beta"+ str(j)
            
            #print " a is" + str(a)
            if ((a-1)*2 % 6 == 0) and a *2 < self._N:
                    conv_param1 = {'stride': 2, 'pad': 1}
                    conv_param2 = {'stride': 1, 'pad': 1}
                    #print "small check"
                    #print a*2
                    l -= 1
                    k = l
                    
            else:
                    conv_param1 = {'stride': 1, 'pad': 1}
                    conv_param2 = {'stride': 1, 'pad': 1}
                    k = 0
            dcw = "CW" + str(k)
            dcb = "Cb" + str(k)
            #print "k is" + str(k)
            
            
            #print 'bachnorm_layer_cache'+i , 'bachnorm_layer_cache'+j
            backwards['resNet_Back_input_'+ str(a)] , grads[iGrad] , grads[iBGrad] , grads[jGrad] , grads[jBGrad] ,grads[gammaGrad],grads[betaGrad] ,grads[gammaGrad2],grads[betaGrad2],grads[dcw],grads[dcb]=  ly.resNetUnit_backward(               backwards['resNet_Back_input_'+f],                                                                                                                    outputs['conv_layer_cache_'+i],                                                                                                                      outputs['relu_cache_'+i],                                                                                                                        outputs['conv_layer_cache_'+j],                                                                                                                                                                                     outputs['relu_cache_'+j] ,                                                                  outputs['bachnorm_layer_cache'+i],                                      outputs['bachnorm_layer_cache'+j],                                                     outputs['x_pad_'+str(k)]  )
                                                                                                                      
                                                                                                                      
                                                                                                                      
            
            
            
            pass
    
        
        ##################################################################################################################
        ##################################################################################################################
        ##############################  loss calulation for classifier     ##############################################
        ################################################################################################################
        ################################################################################################################
        
        
        backwards['batch'], grads['gamma0'],grads['Beta0'] = ly.spatial_batchnorm_backward(backwards['resNet_Back_input_1'],outputs['batch_catch'])
        
        dx,grads['W1'],grads['b1'] = conv_backward_fast ( backwards['batch'], outputs['conv_cache_1'])
        
        #print dx.shape , dw.shape,db.shape , X.shape
        
        #for i in grads:
        #    if grads[i].shape == self.params[i].shape:
        #        #print i + " is ok"
        #        #print grads[i].shape
        #    #else:
        #        #print i + " is not ok"
        #        #print grads[i].shape
        #        pass
        
        #for i in self.params:
            #print i
        
        
        for W_tmp in self.params:
            if W_tmp[0] == 'W':
                grads[W_tmp] += self.reg*self.params[W_tmp]
        
        
        return loss, grads
        
        
        
        pass
