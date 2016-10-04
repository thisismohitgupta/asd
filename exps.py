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
    num_train = 50
    small_data = {
      'X_train': data['X_train'][:num_train],
      'y_train': data['y_train'][:num_train],
      'X_val': data['X_val'],
      'y_val': data['y_val'],
    }


    
    import pickle

    
    res = cnn.ResNet(weight_scale=1e-3,reg=0.5)
    
    infile = open( "params.pickle", "rb" )
    itemlist = pickle.load(infile)
    
    res.params = itemlist
    
    infile.close()
    
    solver = Solver(res, data,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-4,
                      'stride': 1
                    },
                    verbose=True,lr_decay=0.95,
                    num_epochs=2, batch_size=256,
                    print_every=1)

    #os.system("taskset -p 0xff %d" % os.getpid())
    #pool = Pool(4)
    #pool.map(res.loss,small_data['X_train'])
    #print "Fibonacci, parallel: %.3f" %dt
    
    solver.train()
    
    outfile = open( "params.pickle", "wb" )
    pickle.dump( solver.best_params, outfile) 
    #infile = open( "params.pickle", "rb" )
    #itemlist = pickle.load(infile)
    
    #for i in itemlist:
    #    print i
    #    print itemlist[i].shape
    
    
    
