import numpy as np

#----------------#
#                #
#  forward pass  #
#                #
#----------------#
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """  
    input: 
        - x.shape = (N, D)
        - prev_h.shape = (N, H)
        - Wx.shape = (D, H)
        - Wh.shape = (H, H)
        - b.shape = (H,)
    output: 
        - next_h.shape = (N, H)
        - cache = (next_h, x, prev_h, Wx, Wh)
    """
    next_h, cache = None, None

    # compute z and pass through tanh. Save cache
    next_h = np.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (next_h, x, prev_h, Wx, Wh)

    return next_h, cache

#----------------#
#                #
#  backward pass #
#                #
#----------------#
def rnn_step_backward(dnext_h, cache):
    
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    # retrieve values from cache, compute dz with next_h = tanh(z)
    next_h, x, prev_h, Wx, Wh = cache
    dz = dnext_h * (1 - np.square(next_h))

    # compute gradients
    dx = dz @ Wx.T
    dprev_h = dz @ Wh.T
    dWx = x.T @ dz
    dWh = prev_h.T @ dz
    db = dz.sum(axis=0)

    return dx, dprev_h, dWx, dWh, db

#---------------------------------------------#
#                                             #
#  RNN forward on an entire sequence of data  #
#                                             #
#---------------------------------------------#
def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None

    # init args
    cache = []
    h = [h0]

    for t in range(x.shape[1]):
        # run forward pass, retrieve next h and append new cache
        next_h, cache_t = rnn_step_forward(x=x[:, t], 
                                           prev_h=h[t], 
                                           Wx=Wx, 
                                           Wh=Wh, 
                                           b=b)
        h.append(next_h)
        cache.append(cache_t)

    # stack over T, excluding h0
    h = np.stack(h[1:], axis=1)

    return h, cache

#---------------------------------------------#
#                                             #
#  RNN backward on an entire sequence of data #
#                                             #
#---------------------------------------------#
def rnn_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    # get the shape values and initialize gradients
    (N, T, H), (D, _) = dh.shape, cache[0][3].shape
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    
    for t in range(T-1, -1, -1):
        # run backward pass for t^th timestep and update the gradient matrices
        dx_t, dh0, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h=dh[:, t] + dh0, 
                                                          cache=cache[t])
        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    return dx, dh0, dWx, dWh, db

#----------------#
#                #
#  forward pass  #
#                #
#----------------#
def word_embedding_forward(x, W):
    """  
    input:
        - x.shape = (N, T): each element idx of x muxt be in the range 0 <= idx < vocab_size.
            + N: batch_size
            + T: number of tokens
        - W.shape = (vocab_size, embedding_dims)
    output: 
        - out.shape = (N, T, D)
    """
    out, cache = None, None
    
    out, cache = W[x], (x, W)
   
    return out, cache

#----------------#
#                #
#  backward pass #
#                #
#----------------#
def word_embedding_backward(dout, cache):
    """  
    we cannot back-propagate into the words since they are integers, 
    so we only return gradient for the word embedding matrix.
    """
    dW = None

    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW