import numpy as np

from Solver import Solver
from data_utils.data import get_CIFAR10_data
from classifiers import cnn_huge as cnn
import os
from timeit import timeit
from multiprocessing import Pool


if __name__ == '__main__':
    data = get_CIFAR10_data()
    for k, v in data.iteritems():
      print '%s: ' % k, v.shape
    from classifiers.cnn import ThreeLayerConvNet
    num_train = 1000
    small_data = {
      'X_train': data['X_train'][:num_train],
      'y_train': data['y_train'][:num_train],
      'X_val': data['X_val'],
      'y_val': data['y_val'],
    }


    #res = cnn.ResNet(weight_scale=1e-3,reg=0.5)
    #solver = Solver(res, small_data,
    #                update_rule='adam',
    #                optim_config={
    #                  'learning_rate': 1e-4,
    #                  'stride': 1
    #                },
    #                verbose=True,lr_decay=0.95,
    #                num_epochs=20, batch_size=250,
    #                print_every=1)

    res = cnn.ResNet(weight_scale=1.83298071e-03,reg=0.5)
    solver = Solver(res, data,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                      'stride': 1
                    },
                    verbose=True,lr_decay=0.95,
                    num_epochs=50, batch_size=50,
                    print_every=1)

    solver.train()
    #os.system("taskset -p 0xff %d" % os.getpid())
    #pool = Pool(4)
    #pool.map(res.loss,small_data['X_train'])
    #print "Fibonacci, parallel: %.3f" %dt
    
    solver.train()
    
    import pickle
    outfile = open( "params.pickle", "wb" )
    pickle.dump( solver.best_params, outfile) 
    #infile = open( "params.pickle", "rb" )
    #itemlist = pickle.load(infile)
    
    #for i in itemlist:
    #    print i
    #    print itemlist[i].shape
    
    
    
