import numpy as np
from core.rnn_layers import sigmoid

#----------------#
#                #
#  forward pass  #
#                #
#----------------#
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """  
    input: 
        - x.shape = (N, D)
        - prev_h.shape = (N, H)
        - prev_c.shape = (N, H)
        - Wx.shape = (D, 4H)
        - Wh.shape = (H, 4H)
        - b.shape = (4H,)
    output: 
        - next_h.shape = (N, H)
        - next_c.shape = (N, H)
        - cache = (prev_h, prev_c, next_c, i, f, o, g, x, Wx, Wh)
    """
    next_h, next_c, cache = None, None, None
    # compute activations vector A
    a_raw = x @ Wx + prev_h @ Wh + b
    a = np.hsplit(a_raw, 4) # a_raw = np.hstack((a_0, a_1, a_2, a_3))
    
    # split activations into input, forget, output gates and candidate values
    input_gate = a[0]
    forget_gate = a[1]
    output_gate = a[2]
    candidate = a[3] 

    # apply activation function
    i = sigmoid(input_gate)
    f = sigmoid(forget_gate)
    o = sigmoid(output_gate)
    g = np.tanh(candidate)

    # compute next cell state and hidden state
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    # store values for backward pass
    cache = (prev_h, prev_c, next_c, i, f, o, g, x, Wx, Wh)
    return next_h, next_c, cache

#----------------#
#                #
#  backawrd pass #
#                #
#----------------#
def lstm_step_backward(dnext_h, dnext_c, cache):
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
  
    # retrieve elements from cache
    prev_h, prev_c, next_c, i, f, o, g, x, Wx, Wh = cache

    # compute full partial derivative of dnext_c and dprev_c
    dnext_c += dnext_h * o * (1 - np.square(np.tanh(next_c)))
    dprev_c = dnext_c * f

    # partial derivatives w.r.t. a
    da0 = dnext_c * g * i * (1 - i)
    da1 = dnext_c * prev_c * f * (1 - f)
    da2 = dnext_h * np.tanh(next_c) * o * (1 - o)
    da3 = dnext_c * i * (1 - np.square(g))
    da_raw = np.hstack((da0, da1, da2, da3)) # (N, 4H)

    # derivatives w.r.t. primary values
    dx = da_raw @ Wx.T # (N, 4H) @ (4H, D) = (N, D)
    dprev_h = da_raw @ Wh.T # (N, 4H) @ (4H, H) = (N, H)
    dWx = x.T @ da_raw # (D, N) @ (N, 4H) = (D, 4H)
    dWh = prev_h.T @ da_raw # (H, N) @ (N, 4H) = (H, 4H)
    db = da_raw.sum(axis=0) # (4H,)

    return dx, dprev_h, dprev_c, dWx, dWh, db

#---------------------------------------------#
#                                             #
#  LSTM forward on an entire sequence of data #
#                                             #
#---------------------------------------------#
def lstm_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    # init cell, hidden states and cache list
    c, hs, cache = np.zeros_like(h0), [h0], []

    for t in range(x.shape[1]):
        # compute hidden + cell state at timestep t, append cache_t to list
        h, c, cache_t = lstm_step_forward(x[:, t], hs[-1], c, Wx, Wh, b)
        hs.append(h)
        cache.append(cache_t)
    
    # stack along T, excluding h0
    h = np.stack(hs[1:], axis=1)
    return h, cache

#---------------------------------------------#
#                                             #
# LSTM backward on an entire sequence of data #
#                                             #
#---------------------------------------------#
def lstm_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    
    # get the shape values and initialize gradients
    (N, T, H), (D, H4) = dh.shape, cache[0][8].shape
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dc0 = np.zeros((N, H))
    dWx = np.zeros((D, H4))
    dWh = np.zeros((H, H4))
    db = np.zeros(H4)
    
    for t in range(T-1, -1, -1):
        # run backward pass for t^th timestep and update the gradient matrices
        dx_t, dh0, dc0, dWx_t, dWh_t, db_t = lstm_step_backward(dh0 + dh[:, t], dc0, cache[t])
        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    return dx, dh0, dWx, dWh, db
