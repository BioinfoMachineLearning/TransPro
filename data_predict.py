#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:44:07 2021

@author: tianqi
"""
import numpy as np
from typing import Sequence, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import operator,random
from itertools import islice


RawMSA = Sequence[Tuple[str, str]]
proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}

class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for _, seq_str in raw_batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str) in enumerate(raw_batch):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(
                [self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64
            )
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_str)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_str) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return labels, strs, tokens

class MSABatchConverter_bkp(BatchConverter):

    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, :msa_tokens.size(0), :msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens

class MSABatchConverter(BatchConverter):

    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA],batch_labels1 = None,batch_labels2 = None):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        
        labels_de1 = torch.empty(
            (
                batch_size,
                max_seqlen
                + int(True)
                # + int(True),
            ),
            dtype=torch.float32,
        )
        labels_de1.fill_(self.alphabet.padding_idx)
        
        if batch_labels1:
            for i, label in enumerate(batch_labels1):
                #labels[i,0, ] = 0
                labels_de1[i,0,] = 0
                #seq_de = torch.tensor([np.argmax(one_hot)+2 for one_hot in label],dtype=torch.int64)
                seq = torch.tensor(label[0],dtype=torch.float32)
                labels_de1[i,int(True) : len(label[0]) + int(True)] = seq
                #labels_de1[i,0 : len(label[0]) + 0] = seq

        labels_de2 = torch.empty(
            (
                batch_size,
                max_seqlen
                + int(True)
                # + int(True),
            ),
            dtype=torch.float32,
        )
        labels_de2.fill_(self.alphabet.padding_idx)
        
        if batch_labels1:
            for i, label in enumerate(batch_labels1):
                #labels[i,0, ] = 0
                labels_de2[i,0,] = 0
                #seq_de = torch.tensor([np.argmax(one_hot)+2 for one_hot in label],dtype=torch.int64)
                seq = torch.tensor(label[1],dtype=torch.float32)
                #labels_de[i,int(True) : len(label) + int(True)] = seq
                labels_de2[i,int(True) : len(label[1]) + int(True)] = seq

        labels_de3 = torch.empty(
            (
                batch_size,
                max_seqlen
                + int(True)
                # + int(True),
            ),
            dtype=torch.int64,
        )
        labels_de3.fill_(self.alphabet.padding_idx)
        
        if batch_labels2:
            for i, label in enumerate(batch_labels2):
                #labels[i,0, ] = 0
                labels_de3[i,0,] = 0
                seq_de = torch.tensor([np.argmax(one_hot)+2 for one_hot in label],dtype=torch.int64)
                #seq = torch.tensor(label,dtype=torch.float32)
                labels_de3[i,int(True) : len(label) + int(True)] = seq_de

        #labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            #labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, :msa_tokens.size(0), :msa_tokens.size(1)] = msa_tokens

        return labels_de1,labels_de2,labels_de3, strs, tokens

class MyIterator:
    def __init__(self,batch_size,L_lst,train=False):
        self.batch_size = batch_size
        self.L_lst = L_lst
        self.train = train
        self.data = {}
        self.batches = []
        self.create_batches()

    def create_batches(self):
        def sort_lst(L_lst):
            train_dict = {}
            for line in open(L_lst):
                line = line.rstrip()
                arr = line.split()
                train_dict[arr[0]] = int(arr[1])
            train_dict = sorted(train_dict.items(), key=operator.itemgetter(1))
            return dict(train_dict)
    
        def chunks(data,SIZE):
            p_batch = []
            it = iter(data)
            for i in range(0, len(data), SIZE):
                p_batch.append({k:data[k] for k in islice(it, SIZE)})
            return p_batch
        
        train_dict = sort_lst(self.L_lst)
        self.data = train_dict
        if self.train:
            def pool(data):
                batch_block = chunks(data, SIZE=self.batch_size)
                for i in batch_block:
                    p_batch = chunks(i, SIZE=self.batch_size)
                    random.shuffle(p_batch)
                    for b in p_batch: 
                        yield b
            self.batches = pool(self.data)

        else:
            self.batches = []
            batch_block = chunks(self.data, SIZE=self.batch_size)
            for i in batch_block:
                p_batch = chunks(i, SIZE=self.batch_size)
                self.batches.append(p_batch[0])

class Alphabet(object):

    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = False,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return {"toks": self.toks}

    def get_batch_converter(self):
        if self.use_msa:
            return MSABatchConverter(self)
        else:
            return BatchConverter(self)

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(standard_toks=d["toks"], **kwargs)

    @classmethod
    def from_architecture(cls) -> "Alphabet":
        standard_toks = proteinseq_toks["toks"]
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = False
        append_eos = False
        use_msa = True
        return cls(
            standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa
        )

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg1=None,trg2=None,trg3=None, pad=0):
        self.src = src
        self.src_mask = (src[:,0,:] != pad).unsqueeze(-2)
        if trg3 is not None:
            self.trg3 = trg3[:, :-1]
            self.trg3_y = trg3[:, 1:]
        if trg2 is not None:
            self.trg2 = trg2[:, :-1]
            self.trg2_y = trg2[:, 1:]
        if trg1 is not None:
            #trg = src.squeeze(1)
            self.trg1 = trg1[:, :-1]
            self.trg1_y = trg1[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg1[:,:], pad)
            self.ntokens = (self.trg1_y[:,:] != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum() / normalize
            total += l.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

# class SimpleLossCompute:
#     #"A simple loss compute and train function."
#     def __init__(self, generator, criterion, opt=None):
#         self.generator = generator
#         self.criterion = criterion
#         self.opt = opt
        
#     def __call__(self, x, y, norm):
#         x = self.generator(x)
#         loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
#                               y.contiguous().view(-1)) / norm
#         loss.backward()
#         if self.opt is not None:
#             self.opt.step()
#             self.opt.optimizer.zero_grad()
#         return loss.data * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    #ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    ys1 = torch.ones(1, 1,dtype=torch.float32).fill_(start_symbol)
    ys1 = ys1.to(device="cuda:0") 
    ys2 = torch.ones(1, 1,dtype=torch.float32).fill_(start_symbol)
    ys2 = ys2.to(device="cuda:0") 
    ys3 = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    ys3 = ys3.to(device="cuda:0") 
    for i in range(max_len):
        out1,out2,out3 = model.decode(memory, src_mask, 
                           Variable(ys1), Variable(ys2),Variable(ys3),
                           Variable(subsequent_mask(ys1.size(1)).type_as(ys1.data)))
        prob1,prob2 = model.generator1(out1[:, -1],out2[:, -1])
        prob3 = model.generator2(out3[:, -1])
        _, next_word = torch.max(prob3, dim = 1)
        next_word = next_word.data[0]
        ys3 = torch.cat([ys3, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        next_word1 = prob1.data[0]
        next_word1 = next_word1[0]
        next_word2 = prob2.data[0]
        next_word2 = next_word2[0]
        ys1 = torch.cat([ys1, 
                        torch.ones(1, 1).type_as(ys1.data).fill_(next_word1)], dim=1)
        ys2 = torch.cat([ys2, 
                        torch.ones(1, 1).type_as(ys2.data).fill_(next_word2)], dim=1)
    return ys1,ys2,ys3

def greedy_decode2(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys1 = torch.ones(1, 1,dtype=torch.float32).fill_(start_symbol)
    ys1 = ys1.to(device="cuda:0") 
    ys2 = torch.ones(1, 1,dtype=torch.float32).fill_(start_symbol)
    ys2 = ys2.to(device="cuda:0") 
    ys3 = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    ys3 = ys3.to(device="cuda:0")
    
    ys = []
    for i in range(max_len):
        out1,out2,out3 = model.decode(memory, src_mask, 
                           Variable(ys1), Variable(ys2),Variable(ys3),
                           Variable(subsequent_mask(ys1.size(1)).type_as(ys1.data)))
        prob1,prob2 = model.generator1(out1[:, -1],out2[:, -1])
        prob3 = model.generator2(out3[:, -1])
        _, next_word = torch.max(prob3, dim = 1)
        next_word = next_word.data[0]
        ys3 = torch.cat([ys3, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        next_word1 = prob1.data[0]
        next_word1 = next_word1[0]
        next_word2 = prob2.data[0]
        next_word2 = next_word2[0]
        ys1 = torch.cat([ys1, 
                        torch.ones(1, 1).type_as(ys1.data).fill_(next_word1)], dim=1)
        ys2 = torch.cat([ys2, 
                        torch.ones(1, 1).type_as(ys2.data).fill_(next_word2)], dim=1)
        ys.append(prob3.cpu().tolist()[0])
    return ys3,ys