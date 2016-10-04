
import numpy as np



"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w, giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w += -config['learning_rate'] * dw

    return w, config
def adam( x, dx, config=None):
    # adagrad + momentum
    # invented in 2014 itself

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.995)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config.                                                         #
    #############################################################################

    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']

    config['t'] += 1

    #blow line resembles momentum update


    config['m'] = beta1 * config['m'] + (1 - beta1) * dx

    #below line resembles rms prop
    config['v'] = beta2 * config['v'] + (1 - beta2) * dx ** 2

    # below is the bias correction and is only relevant for a few steps
    mt_hat = config['m'] / (1 - (beta1) ** config['t'])
    vt_hat = config['v'] / (1 - (beta2) ** config['t'])

    next_x = x - learning_rate * mt_hat / (np.sqrt(vt_hat + epsilon))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config