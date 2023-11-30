import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#-------------------------------------------------------------------------#
# this file defines layer types that are commonly used for transformers.  #
# - PositionalEncoding                                                    #
# - MultiheadAttention                                                    #
#-------------------------------------------------------------------------#


class PositionalEncoding(nn.Module):
    """
    encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        inputs:
          - embed_dim: the size of the embed dimension
          - dropout: the dropout value
          - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # create an array with a "batch dimension" of 1 (which will broadcast across all examples in the batch)
        pe = torch.zeros(1, max_len, embed_dim)
        
        # get col idx range (i) and powers
        i = torch.arange(max_len)[:, None]
        pows = torch.pow(10000, -torch.arange(0, embed_dim, 2) / embed_dim)

        # compute positional values sin/cos
        pe[0, :, 0::2] = torch.sin(i * pows)
        pe[0, :, 1::2] = torch.cos(i * pows)

        # make sure the positional encodings will be saved with the model parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        element-wise add positional embeddings to the input sequence.

        inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim

        output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
    
        output = x + self.pe[:, :S]
        output = self.dropout(output)        
        return output


class MultiHeadAttention(nn.Module):
    """
    a model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        inputs:
          - embed_dim: Dimension of the token embedding
          - num_heads: Number of attention heads
          - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        """ 
        we will initialize these layers for you, since swapping the ordering
        would affect the random number generation (and therefore your exact
        outputs relative to the autograder). Note that the layers use a bias
        term, but this isn't strictly necessary (and varies by
        implementation). 
        """
        
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
      
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(embed_dim / num_heads)
        
    def forward(self, query, key, value, attn_mask=None):
        """
        calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        in the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        inputs:
          - query: Input data to be used as the query, of shape (N, S, E)
          - key: Input data to be used as the key, of shape (N, T, E)
          - value: Input data to be used as the value, of shape (N, T, E)
          - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates token
            i in the target should not be influenced by token j in the source.
            
        output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, D = query.shape
        N, T, D = value.shape
        # create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, T, D))
        
        """         
        step-by-step                                                            
          1) you'll want to split your shape from (N, T, E) into (N, T, H, E/H),  
              where H is the number of heads.                                      
          2) the function torch.matmul allows you to do a batched matrix multiply.
              For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  
              shape (N, H, T, T). For more examples, see                           
              https://pytorch.org/docs/stable/generated/torch.matmul.html          
          3) for applying attn_mask, think how the scores should be modified to   
              prevent a value from influencing output. Specifically, the PyTorch   
              function masked_fill may come in handy.                              
        """        
        # get num of heads
        H = self.num_heads

        # compute key, query and value matrices from sequences
        K = self.key(key).view(N, T, H, D//H).moveaxis(1, 2)
        Q = self.query(query).view(N, S, H, D//H).moveaxis(1, 2)
        V = self.value(value).view(N, T, H, D//H).moveaxis(1, 2)

        # (N,H,S,D/H) @ (N,H,D/H,T) -> (N,H,S,T)
        Y = Q @ K.transpose(2, 3) / self.scale

        if attn_mask is not None:
            # ensure small probabilities in softmax
            Y = Y.masked_fill(attn_mask==0, float("-inf"))
      
        # (N,H,S,T) @ (N,H,T,D/H) -> (N,H,S,D/H)
        Y = self.dropout(F.softmax(Y, dim=-1)) @ V
        output = self.proj(Y.moveaxis(1, 2).reshape(N, S, D))

        return output


