
import numpy as np
class Checks:
    """


    """

    def numerical_gradient(self,x,w,b,method,dx):

        return ((method(x+1e-7,w,b)[0] - (method(x,w,b) )[0]) ) * dx/ 1e-7

    def eval_numerical_gradient_array( self,x,f ,df, h=1e-5):
        """
        Evaluate a numeric gradient for a function that accepts a numpy
        array and returns a numpy array.
        """
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index

            oldval = x[ix]
            x[ix] = oldval + h
            pos = f(x).copy()
            x[ix] = oldval - h
            neg = f(x).copy()
            x[ix] = oldval

            grad[ix] = np.sum((pos - neg) * df) / (2 * h)
            it.iternext()
        return grad
