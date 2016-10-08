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

    data['X_train'] -= data['X_train'].mean(axis=0)
    data['X_val'] -= data['X_train'].mean(axis=0)

    res = cnn.ResNet(weight_scale=5.4e-02,  reg=0.00005)
    
    infile = open( "best_params.pickle", "rb" )
    itemlist = pickle.load(infile)
    

    infile = open( "best_batch_params.pickle", "rb" )
    itemlista = pickle.load(infile)


    res.params = itemlist
    res.bn_params = itemlista
    infile.close()
    
    solver = Solver(res, data,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                      'stride': 1
                    },
                    verbose=True,lr_decay=0.1,
                    num_epochs=25, batch_size=64,
                    print_every=1)

    #os.system("taskset -p 0xff %d" % os.getpid())
    #pool = Pool(4)
    #pool.map(res.loss,small_data['X_train'])
    #print "Fibonacci, parallel: %.3f" %dt
    
    solver.train()
    
    outfile = open( "best_params.pickle", "wb" )
    pickle.dump( solver.best_params, outfile)
    outfile.close()
    
    outfile = open( "best_batch_params.pickle", "wb" )
    pickle.dump( res.bn_params, outfile)
    outfile.close()


    outfile = open("best_val_acc.pickle", "wb")
    pickle.dump(solver.best_val_acc, outfile)
    outfile.close()



    outfile = open("loss_history.pickle", "wb")
    pickle.dump(solver.loss_history, outfile)
    outfile.close()

    outfile = open("train_acc_history.pickle", "wb")
    pickle.dump(solver.train_acc_history, outfile)
    outfile.close()

    #best_val_acc
    outfile = open("val_acc_history.pickle", "wb")
    pickle.dump(solver.val_acc_history, outfile)
    outfile.close()

    #infile = open( "params.pickle", "rb" )
    #itemlist = pickle.load(infile)
    
    #for i in itemlist:
    #    print i
    #    print itemlist[i].shape
