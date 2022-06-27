#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:46:53 2021

@author: tianqi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from torch.autograd import Variable

import math, copy, time

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed1,tgt_embed2, generator1, generator2):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed1 = tgt_embed1
        self.tgt_embed2 = tgt_embed2
        self.generator1 = generator1
        self.generator2 = generator2
        
    def forward(self, src, tgt1, tgt2, tgt3, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,tgt1,tgt2, tgt3, tgt_mask)
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt1, tgt2, tgt3, tgt_mask):
        tgt1 = tgt1.unsqueeze(-1)
        tgt2 = tgt2.unsqueeze(-1)
        return self.decoder(self.tgt_embed1(tgt1), memory, src_mask, tgt_mask),self.decoder(self.tgt_embed1(tgt2),memory, src_mask, tgt_mask), self.decoder(self.tgt_embed2(tgt3),memory, src_mask, tgt_mask)


class SimpleMLP(nn.Module):
    def __init__(self, d_model,hid_dim,out_dim,dropout):
        super(SimpleMLP, self).__init__()
        self.proj1 = weight_norm(nn.Linear(d_model, hid_dim), dim=None)
        self.proj3 = weight_norm(nn.Linear(d_model, hid_dim), dim=None)
        self.proj2 = weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x1,x2):
        x1 = self.proj2(self.dropout(F.tanh(self.proj1(x1))))
        x2 = self.proj2(self.dropout(F.tanh(self.proj3(x2))))
        return x1,x2
    
class Generator(nn.Module):
    #"Define standard linear + softmax generation step."
    def __init__(self, d_model,hid_dim, vocab):
        super(Generator, self).__init__()
        self.proj1 = nn.Linear(d_model, hid_dim)
        self.proj = nn.Linear(hid_dim, vocab)

    def forward(self, x):
        x = self.proj1(x)
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Embeddings_src(nn.Module):
    #"Define the input Embedding for variable N of MSA."
    def __init__(self, src_vocab = 33, fn_dim =8, padding_idx = 1, max_positions = 1500,embed_dim=64):
        super( Embeddings_src, self ).__init__() 
        self.src_vocab = src_vocab
        self.embed_dim = embed_dim
        self.fn_dim = fn_dim
        self.padding_idx = padding_idx
        self.max_positions = max_positions
        self.embed_tokens = nn.Embedding(self.src_vocab, self.embed_dim, padding_idx=self.padding_idx)
        self.embed_positions = LearnedPositionalEmbedding(self.max_positions, self.embed_dim, self.padding_idx,)
        self.msa_position_embedding = nn.Parameter(0.01 * torch.randn(1, self.max_positions, 1, 1),requires_grad=True,)
        self.fc1 = nn.Linear(self.embed_dim ,self.fn_dim )
        
    def forward(self, src):
        ##Input layerc
        tokens = src
        batch_size, num_alignments, seqlen = tokens.size()
 
        ##embed_tokens
        x = self.embed_tokens(tokens)
        #print("Input after embedding ....{}".format(x.size()))
        ##embed_tokens + embed_positions
        #x += self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())
        ##embed_tokens + embed_positions + msa_position_embedding
        #x += self.msa_position_embedding[:, :num_alignments]
        #print("Row + column position embedding ....{}".format(x.size()))
        ##columnwise attention for 1d input
   
        input = x.permute(0, 2, 1, 3)
        input_p = input.permute(0, 1, 3, 2)
        #print(input_p.size())
        x = self.fc1(input)
        x = F.softmax(x, dim = -2)
        #print(x.size())
        x = torch.matmul(input_p,x)
        #print(x.size())
        x = x.view(batch_size,seqlen,self.embed_dim*self.fn_dim)
        #print("Column attention ....{}".format(x.size()))  
        return x

class NoamOpt:
    #"Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        #"Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        #"Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    #"Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class MSE(nn.Module):
    #"Implement label smoothing."
    def __init__(self, padding_idx):
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss(reduction = 'sum')
        self.padding_idx = padding_idx
        self.true_dist = None
        
    def forward(self, x, target):
        true_dist = target.data.clone()
        true_dist = true_dist.unsqueeze(1)
        #true_dist.fill_(self.smoothing / (self.size - 2))
        #true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    #"A simple loss compute and train function."    
    def __init__(self, generator1,generator2, criterion1, criterion2, opt=None):
        self.generator1 = generator1
        self.generator2 = generator2
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.opt = opt
        
    def __call__(self, x1,x2,x3, y1,y2,y3, norm):
        x1 = x1.to(device="cuda:0")
        x2 = x2.to(device="cuda:0")
        x3 = x3.to(device="cuda:0")
        y1 = y1.to(device="cuda:0") 
        y2 = y2.to(device="cuda:0")
        y3 = y3.to(device="cuda:0")
        x1,x2 = self.generator1(x1,x2)
        x3 = self.generator2(x3)
        loss1 = self.criterion1(x1.contiguous().view(-1, x1.size(-1)), 
                              y1.contiguous().view(-1))
        loss2 = self.criterion1(x2.contiguous().view(-1, x2.size(-1)), 
                              y2.contiguous().view(-1))
        loss3 = self.criterion2(x3.contiguous().view(-1, x3.size(-1)), 
                              y3.contiguous().view(-1))
        #print(0.4*loss1,0.4*loss2,0.2*loss3)
        loss = (2*loss1+2*loss2+loss3)/norm
        loss.backward()
        
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

def run_epoch(data_iter, model, loss_compute):
    #"Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    result = 0
    for i, batch in enumerate(data_iter):
        out1,out2,out3 = model.forward(batch.src, batch.trg1, batch.trg2, batch.trg3,
                            batch.src_mask, batch.trg_mask)
        loss= loss_compute(out1,out2,out3, batch.trg1_y,batch.trg2_y,batch.trg3_y, batch.ntokens*2)
        total_loss += loss
        total_tokens += batch.ntokens*2
        tokens += batch.ntokens*2
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / (batch.ntokens*2), tokens / elapsed))
            start = time.time()
            tokens = 0
        result = total_loss/total_tokens
    return result


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=48, h=8, dropout=0.1, max_positions=1500, padding_idx=1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings_src(src_vocab, int(d_model/64), padding_idx, max_positions), c(position)),
        nn.Sequential(nn.Linear(1,d_model), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        SimpleMLP(d_model, 64, 1, dropout),
        Generator(d_model, 64, tgt_vocab))  #Generator(d_model, tgt_vocab)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model