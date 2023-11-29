import numpy as np


#------------------------------#
#                              #
# Stochastic gradient descent  #
#                              #
#------------------------------#
def sgd(w, dw, config=None):
    """
    config format:
      - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

#------------------------------#
#                              #
#     SGD with Momentum        #
#                              #
#------------------------------#
def sgd_momentum(w, dw, config=None):
    """
    config format:
      - learning_rate: Scalar learning rate.
      - momentum: Scalar between 0 and 1 giving the momentum value.
        Setting momentum = 0 reduces to sgd.
      - velocity: A numpy array of the same shape as w and dw used to store a
        moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    v *= config["momentum"]
    v -= config["learning_rate"] * dw
    w += v
    next_w = w
    config["velocity"] = v

    return next_w, config

#------------------------------#
#                              #
# Root Mean Square Propagation #
#                              #
#------------------------------#
def rmsprop(w, dw, config=None):
    """
    uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
      - learning_rate: Scalar learning rate.
      - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
        gradient cache.
      - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    rho = config["decay_rate"]
    lr = config["learning_rate"]
    eps = config["epsilon"]
    config["cache"] *= rho
    config["cache"] += (1.0 - rho) * dw ** 2
    step = -(lr * dw) / (np.sqrt(config["cache"]) + eps)
    w += step
    next_w = w

    return next_w, config


#------------------------------#
#                              #
#  adaptive moment estimation  #
#                              #
#------------------------------#
def adam(w, dw, config=None):
    """
    uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
      - learning_rate: Scalar learning rate.
      - beta1: Decay rate for moving average of first moment of gradient.
      - beta2: Decay rate for moving average of second moment of gradient.
      - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      - m: Moving average of gradient.
      - v: Moving average of squared gradient.
      - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    beta1, beta2, eps = config["beta1"], config["beta2"], config["epsilon"]
    t, m, v = config["t"], config["m"], config["v"]
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw * dw)
    t += 1
    alpha = config["learning_rate"] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    w -= alpha * (m / (np.sqrt(v) + eps))
    config["t"] = t
    config["m"] = m
    config["v"] = v
    next_w = w

    return next_w, config
