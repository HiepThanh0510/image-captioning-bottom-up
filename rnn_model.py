import numpy as np

from core.rnn_layers import *

from rnn_utils import *
from lstm_utils import *

class CaptioningRNN:
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128, cell_type="rnn", dtype=np.float32):
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # initialize word vectors
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = np.zeros(hidden_dim)

        # initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # initialize output to vocab weights
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        # cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    
    
    
    def loss(self, features, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = captions_out != self._null

        # weight and bias for the affine transform from image features to initial hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # word embedding matrix
        W_embed = self.params["W_embed"]

        # input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss, grads = 0.0, {}
        
        if self.cell_type == "rnn":
            # if cell type is regular RNN
            recurrent_forward = rnn_forward
            recurrent_backward = rnn_backward
        elif self.cell_type == "lstm":
            # if cell type is long short-term
            recurrent_forward = lstm_forward
            recurrent_backward = lstm_backward
            
        # note about the step by step
        # forward passes
        # (1) use an affine transformation to compute the initial hidden state
        #     from the image features. This should produce an array of shape (N, H)
        h0, cache_h0 = affine_forward(features, W_proj, b_proj)

        # (2) use a word embedding layer to transform the words in captions_in
        #     from indices to vectors, giving an array of shape (N, T, W).
        x, cache_x = word_embedding_forward(captions_in, W_embed)

        # (3) use either a vanilla RNN or LSTM (depending on self.cell_type) to
        #     process the sequence of input word vectors and produce hidden state
        #     vectors for all timesteps, producing an array of shape (N, T, H).
        h, cache_h = recurrent_forward(x, h0, Wx, Wh, b)

        # (4) use a (temporal) affine transformation to compute scores over the
        #     vocabulary at every timestep using the hidden states, giving an
        #     array of shape (N, T, V).
        out, cache_out = temporal_affine_forward(h, W_vocab, b_vocab)

        # (5) use (temporal) softmax to compute loss using captions_out, ignoring
        #     the points where the output word is <NULL> using the mask above.
        loss, dout = temporal_softmax_loss(out, captions_out, mask)

        # backward passes
        dout, dW_vocab, db_vocab = temporal_affine_backward(dout, cache_out)
        dout, dh0, dWx, dWh, db = recurrent_backward(dout, cache_h)
        dW_embed = word_embedding_backward(dout, cache_x)
        _, dW_proj, db_proj = affine_backward(dh0, cache_h0)
        
        grads = {
            "W_proj": dW_proj,
            "b_proj": db_proj,
            "W_embed": dW_embed,
            "Wx": dWx,
            "Wh": dWh,
            "b": db,
            "W_vocab": dW_vocab,
            "b_vocab": db_vocab
        }
        return loss, grads

    def save_checkpoint(self, filename):
        checkpoint = {'W_embed': self.params['W_embed'],
                      'W_proj': self.params['W_proj'],
                      'b_proj': self.params['b_proj'],
                      'Wx': self.params['Wx'],
                      'Wh': self.params['Wh'],
                      'b': self.params['b'],
                      'W_vocab': self.params['W_vocab'],
                      'b_vocab': self.params['b_vocab']}
        
        np.savez(filename, **checkpoint)
        print("checkpoint is saved successfully!")
    
    def load_checkpoint(self, filename):
        checkpoint = np.load(file=filename,
                             allow_pickle=True)
        
        self.params['W_embed'] = checkpoint['W_embed']
        self.params['W_proj'] = checkpoint['W_proj']
        self.params['b_proj'] = checkpoint['b_proj']
        self.params['Wx'] = checkpoint['Wx']
        self.params['Wh'] = checkpoint['Wh']
        self.params['b'] = checkpoint['b']
        self.params['W_vocab'] = checkpoint['W_vocab']
        self.params['b_vocab'] = checkpoint['b_vocab']
        
        """  
        # another way
        list_params = list(dict(checkpoint))
        for param in list_params:
            self.params[param] = checkpoint[param]
        """
        
        print("checkpoint is loaded successfully!")
        
    
    def sample(self, features, max_length=30):
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        # initialize the hidden and cell states, input
        h, _ = affine_forward(features, W_proj, b_proj)
        x = np.repeat(self._start, N)
        c = np.zeros_like(h)

        for t in range(max_length):
            # generate the word embedding of a previous word
            x, _ = word_embedding_forward(x, W_embed)

            if self.cell_type == "rnn":
                # if cell type is regular RNN
                h, _ = rnn_step_forward(x, h, Wx, Wh, b)
            elif self.cell_type == "lstm":
                # if cell type is long short-term memory
                h, c, _ = lstm_step_forward(x, h, c, Wx, Wh, b)

            # compue the final forward pass for t to get scores
            out, _ = affine_forward(h, W_vocab, b_vocab)
            x = np.argmax(out, axis=1)
            captions[:, t] = x

        return captions