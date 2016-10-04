
import numpy as np

from Solver import Solver
from data_utils.data import get_CIFAR10_data

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape
from classifiers.cnn import ThreeLayerConvNet
num_train = 4
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}


from classifiers import cnn_huge as cnn

res = cnn.ResNet()
res.loss(data['X_train'][:num_train])

# x = data['X_train'][:1]
# w = np.random.randn(16, 16, 3, 3)
# w2 = np.random.randn(16, 16, 3, 3)
# b = np.random.randn(16,)
# b2 = np.random.randn(16,)
# #dout = np.random.randn(2, 3, 8, 8)
# conv_param = {'stride': 1, 'pad': 1}
#
#
#
#
# import Layers as layers
# ly = layers.Layers()
# out , chache3 = ly.conv_forward_naive(x,np.random.randn(16, 3, 3, 3),np.random.randn(16,),{'stride': 1, 'pad': 1})
#
# print out.shape
#
# conv_out,cache1, rcache, cache2, rcahce2, x_pad = ly.resNetUnit_forward(out,w,b,w2,b2,conv_param)
#
# print conv_out.shape
#
# dx,dw1 = ly.resNetUnit_backward(conv_out,cache1, rcache, cache2, rcahce2)
#
# print dx.shape
#
# import checks as cks
#
# hello = cks.Checks()
#
# grad = hello.eval_numerical_gradient_array( out,lambda X: ly.resNetUnit_forward(X,w,b,w2,b2,conv_param)[0], dx)
#
#
# #print grad[1].shape
# #print np.subtract(grad, dw1)
# print np.sum(np.subtract(grad, dx))

# model = ThreeLayerConvNet(weight_scale=1e-2)
#
# solver = Solver(model, small_data,
#                 num_epochs=10, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-3,
#                 },
#                 verbose=True, print_every=1)
# solver.train()
#
#
# import matplotlib.pyplot as plt
# plt.subplot(2, 1, 1)
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('iteration')
# plt.ylabel('loss')
#
# plt.subplot(2, 1, 2)
# plt.plot(solver.train_acc_history, '-o')
# plt.plot(solver.val_acc_history, '-o')
# plt.legend(['train', 'val'], loc='upper left')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()

# model = ThreeLayerConvNet()
#
# N = 50
# X = np.random.randn(N, 3, 32, 32)
# y = np.random.randint(10, size=N)
#
# loss, grads = model.loss(X, y)
# print 'Initial loss (no regularization): ', loss
#
# model.reg = 0.5
# loss, grads = model.loss(X, y)
# print 'Initial loss (with regularization): ', loss
#
# num_inputs = 2
# input_dim = (3, 16, 16)
# reg = 0.0
# num_classes = 10
# X = np.random.randn(num_inputs, *input_dim)
# y = np.random.randint(num_classes, size=num_inputs)
#
#
# from checks import Checks
#
# ck = Checks()
# model = ThreeLayerConvNet(num_filters=3, filter_size=3,
#                           input_dim=input_dim, hidden_dim=7,
#                           dtype=np.float64)
# loss, grads = model.loss(X, y)
# for param_name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#
#
#     param_grad_num = ck.eval_numerical_gradient_array( model.params[param_name],f,grads[param_name])
#     e = np.sum(np.subtract(param_grad_num, grads[param_name]))
#     print '%s max relative error: %e' % (param_name, np.sum(np.subtract(param_grad_num, grads[param_name])))
#
# x = np.random.randn(2, 3, 8, 8)
# w = np.random.randn(3, 3, 3, 3)
# b = np.random.randn(3,)
# dout = np.random.randn(2, 3, 8, 8)
# conv_param = {'stride': 1, 'pad': 1}
#
#
# from Layers import Layers
# ly = Layers()
#
# x_shape = (2, 3, 4, 4)
# x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
# pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
#
# out, _ = ly.max_pool_forward_naive(x, pool_param)
#
# correct_out = np.array([[[[-0.26315789, -0.24842105],
#                           [-0.20421053, -0.18947368]],
#                          [[-0.14526316, -0.13052632],
#                           [-0.08631579, -0.07157895]],
#                          [[-0.02736842, -0.01263158],
#                           [ 0.03157895,  0.04631579]]],
#                         [[[ 0.09052632,  0.10526316],
#                           [ 0.14947368,  0.16421053]],
#                          [[ 0.20842105,  0.22315789],
#                           [ 0.26736842,  0.28210526]],
#                          [[ 0.32631579,  0.34105263],
#                           [ 0.38526316,  0.4       ]]]])
#
# # Compare your output with ours. Difference should be around 1e-8.
# print 'Testing max_pool_forward_naive function:'
# print 'difference: ', np.sum(np.subtract(out, correct_out))

#
# out, cache = ly.conv_relu_forward(x, w, b, conv_param)
# dx, dw, db = ly.conv_relu_backward(dout, cache)
#
# from checks import Checks
#
# ck = Checks()
# dx_num = ck.eval_numerical_gradient_array(lambda x: ly.conv_relu_forward(x, w, b, conv_param)[0], x, dout)
# dw_num = ck.eval_numerical_gradient_array(lambda w: ly.conv_relu_forward(x, w, b, conv_param)[0], w, dout)
# db_num = ck.eval_numerical_gradient_array(lambda b: ly.conv_relu_forward(x, w, b, conv_param)[0], b, dout)
#
# print 'Testing conv_relu:'
# print 'dx error: ', np.subtract(dx_num, dx)
# print 'dw error: ', np.subtract(dw_num, dw)
# print 'db error: ', np.subtract(db_num, db)


from Layers import Layers
from checks import Checks

# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# X = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)
# gamma = np.random.rand(20)
#
# beta = np.random.randn(20)
#
# param = {}
# param["mode"]= 'train'
# param['p'] = 0.5
# conv_param = {}
# conv_param["stride"] = 2
# conv_param["pad"] = 1
#
# layers = Layers()
# convolve_layer,cout = layers.conv_forward_naive(X, w, b , conv_param )
# correct_out = np.array([[[[[-0.08759809, -0.10987781],
#                            [-0.18387192, -0.2109216]],
#                           [[0.21027089, 0.21661097],
#                            [0.22847626, 0.23004637]],
#                           [[0.50813986, 0.54309974],
#                            [0.64082444, 0.67101435]]],
#                          [[[-0.98053589, -1.03143541],
#                            [-1.19128892, -1.24695841]],
#                           [[0.69108355, 0.66880383],
#                            [0.59480972, 0.56776003]],
#                           [[2.36270298, 2.36904306],
#                            [2.38090835, 2.38247847]]]]])
#
# print np.sum(correct_out - convolve_layer)
#
# relu_f, cache2 = layers.relu_forward(convolve_layer)
#
# w1 = np.random.normal(scale=1e-3, size=(3*4*4/4, 20))
# b1 = 20
# out, cache = layers.affine_foward(relu_f, w1, b1)
#
# dropoutss, cache4 = layers.dropout_forward(out, param)
#
# batch_norm,cache3 = layers.batchnorm_forward(dropoutss,gamma,beta,param)
#
#
# dconv = layers.conv_backward_naive(convolve_layer,cout)
#
#
# # dx, dw, db = layers.affine_backward(out, cache)
# # relu_b = layers.relu_backward(relu_f, cache2)
# #
# # batch_back,deltaGamma,deltaBeta = layers.batchnorm_backward_alt(batch_norm,cache3)
# # ddropout = layers.dropout_backward(dropoutss,cache4)
#
#
#
#
# check = Checks()
# print np.sum(dconv[0] - check.eval_numerical_gradient_array(X,lambda X: layers.conv_forward_naive(X, w, b , conv_param )[0],convolve_layer))
# #
# # print "affine backward check"
# # print np.sum(dx - check.eval_numerical_gradient_array(X,lambda x: layers.affine_foward(x,w,b)[0],out))
# # print np.sum(dw - check.eval_numerical_gradient_array(w,lambda x: layers.affine_foward(X,w,b)[0],out))
# # print np.sum(db - check.eval_numerical_gradient_array(b,lambda x: layers.affine_foward(X,w,b)[0],out))
# # print "Relu backward check"
# # print np.sum(relu_b - check.eval_numerical_gradient_array(out,lambda x: layers.relu_forward(x)[0],relu_f))
# # print "batch norm bacward check "
# # print np.sum(deltaBeta - check.eval_numerical_gradient_array(beta,lambda x: layers.batchnorm_forward(relu_f,gamma,beta,param)[0],batch_norm))
# # print np.sum(deltaGamma - check.eval_numerical_gradient_array(gamma,lambda x: layers.batchnorm_forward(relu_f,gamma,beta,param)[0],batch_norm))
# # print np.sum(batch_back - check.eval_numerical_gradient_array(relu_f,lambda x: layers.batchnorm_forward(relu_f,gamma,beta,param)[0],batch_norm))
# # print "Dropout check"
# # print np.sum(ddropout - check.eval_numerical_gradient_array(relu_f,lambda xx: layers.dropout_forward(xx,param)[0],dropoutss))
# #
#
#
# #print ddropout
# #print check.eval_numerical_gradient_array(relu_f,lambda x: layers.dropout_forward(relu_f,param)[0],dropoutss)
# #print check.eval_numerical_gradient_array(lambda beta: layers.batchnorm_forward(relu_f,gamma,beta,param)[0],beta,batch_norm).shape