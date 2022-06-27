#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from Bio import SeqIO
import itertools
import string

from data import *
from modules import *

import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

# def read_msa(filename: str) -> List[Tuple[str, str]]:
    # """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    # return [(record.description, remove_insertions(str(record.seq)))
            # for record in SeqIO.parse(filename, "fasta")]

def read_label(filename: str):
    label = np.load(filename)
    return label

def rebatch(b,aln_type,pad_idx = 1,max_positions=500):
    train_msa = []
    tor_label = []
    ss_label = []
    for t in list(b):
        if not os.path.exists(train_dir+"/a3m/"+aln_type+"/"+t+".a3m"):
            continue
        if not os.path.exists(train_dir+"/phi_psi/"+t+".npy"):
            continue
        if not os.path.exists(train_dir+"/ss/ss_3/"+t+".npy"):
            continue
        train_msa.append(read_msa(train_dir+"/a3m/"+aln_type+"/"+t+".a3m",max_positions))
        tor_label.append(read_label(train_dir+"/phi_psi/"+t+".npy"))
        ss_label.append(read_label(train_dir+"/ss/ss_3/"+t+".npy"))
    #- . <cls> <eos> <mask> <null_1> <pad> <unk> A B C D E F G H I K L M N O P Q R S T U V W X Y Z
    #30 29 0 2 32 31 1 3 5 25 23 13 9 18 6 21 12 15 4 20 17 28 14 16 10 8 11 26 7 22 24 19 27
  
    msa_alphabet = Alphabet.from_architecture()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    msa_batch_labels_de1,msa_batch_labels_de2,msa_batch_labels_de3,msa_batch_strs, msa_batch_tokens = msa_batch_converter(train_msa,tor_label,ss_label)
    return Batch(msa_batch_tokens, msa_batch_labels_de1,msa_batch_labels_de2,msa_batch_labels_de3,pad_idx)

def TGT_itos_3(index):
    index = int(index)
    if index == 0:
        tgt = "<s>"
    elif index == 1:
        tgt = "<pad>"
    elif index == 2:
        tgt = "H"
    elif index == 3:
        tgt = "E"
    elif index == 4:
        tgt = "C"
    elif index == 5:
        tgt = "</s>"
    else:
        tgt = "<unk>"
    return tgt

def cal_q3(pred,target):
    count = 0.0
    for i in range(len(target)):
        if pred[i] == target[i]:
            count = count +1
    return count/len(target)

# GPUs to use
devices = [0]
pad_idx = 1
padding_idx=1
SRC_vocab = 33
TGT_vocab = 5
N=6
d_model=512
d_ff=2048
h=8
dropout=0.1
max_positions=1500

src_dir = os.path.dirname(os.path.abspath(__file__))

model = make_model(SRC_vocab, TGT_vocab, N, d_model, d_ff, h, dropout, max_positions, pad_idx)

criterion1 = MSE(padding_idx=padding_idx)
criterion1.cuda()

criterion2 = LabelSmoothing(size=TGT_vocab, padding_idx=padding_idx, smoothing=0.1)
criterion2.cuda()

BATCH_SIZE =30
data_dir = "/storage/htc/bdm/tianqi/test/3d/test"
dataset = "casp14"

train_dir = os.path.join(data_dir, dataset)
test_lst = os.path.join(train_dir, "test.lst")

############################Prediction##########################
q3_lst = []
aln_types = ["bfd","deepmsa"]
#model_dir = ["model/model_current_a3m3","model/model_current_a3m4","model/model_current_a3m1","model/model_current_a3m9","model/model_current_a3m11","model/model_current_a3m12"]
model_dir = ["model/model_current_a3m3","model/model_current_a3m4","model/model_current_a3m1","model/model_current_a3m9","model/model_current_a3m11"]

test_iter = MyIterator(batch_size=1, L_lst=test_lst ,train=False)

for i, batch_orig in enumerate(test_iter.batches):
    out = []
    start = time.time()
    tar = list(batch_orig.keys())[0] 
    if not os.path.exists(train_dir+"/ss/ss_3/"+tar+".npy"):
        #print(train_dir+"/aln/a3m/"+tar+".a3m doesn't exist....")
        continue
    if not os.path.exists(train_dir+"/phi_psi/"+tar+".npy"):
        #print(train_dir+"/aln/a3m/"+tar+".a3m doesn't exist....")
        continue

    for aln_type in aln_types:
        batch = rebatch(batch_orig,aln_type, pad_idx,max_positions)
        src_mask = batch.src_mask
        src = batch.src
        src = batch.src.to(device="cuda:0")
        batch_size, n, seq_l = src.size()
        src_mask = src_mask.to(device="cuda:0")
        
        for model_file in model_dir:
            if os.path.exists(model_file):
                model = torch.load(model_file)
            
            model_par = nn.DataParallel(model, device_ids=devices)
            #model_par = model
            
            model_par.eval()
            out3,out3_prob = greedy_decode2(model, src, src_mask, max_len=seq_l, start_symbol=0)
        
            out.append(out3_prob)
        
    a = np.array(out)
    out = np.average(a, axis=0)
    #out = 5*(a[0]+a[1]+a[2])+3*(a[3]+a[4]+a[5])+(a[6]+a[7]+a[8])+(a[9]+a[10]+a[11])+(a[12]+a[13]+a[14])+(a[15]+a[16]+a[17])+(a[18]+a[19]+a[20])+(a[21]+a[22]+a[23])+(a[24]+a[25]+a[26])
    b = np.argmax(out, axis=1)
    out3_lst = b.tolist()

    #out3_lst = out3[0,1:].cpu().tolist()
    trg_y3 = batch.trg3_y
    trg_y3_lst = trg_y3[0].tolist()

    fline=open(train_dir+"/index/"+tar+".idx").readline().rstrip()
    print("Translation:", end="\t")
    sym = ""
    pred = ""
    pred_lst = []
    sym_lst = []
    for i in range(1, out3.size(1)):
        sym += TGT_itos_3(out3[0, i])
        if sym == "</s>": break
    test_idx = fline.split(',')
    test_idx = [int(i) for i in test_idx]
    for i in test_idx:
        print(sym[i],end="")
        pred += sym[i]
        pred_lst.append(out3_lst[i])
    print()

    print("Target:", end="\t")
    sym = ""
    for i in range(0, batch.trg3_y.size(1)):
        if batch.trg3_y[0,i] == 1: break
        sym += TGT_itos_3(batch.trg3_y[0,i])
        sym_lst.append(trg_y3_lst[i])
    print(sym)
    
    #q3 = cal_q3(pred,sym)
    print(tar)
    q3 = accuracy_score(sym_lst, pred_lst)
    print("{} Q3:{}".format(tar,q3))
    q3_lst.append(q3)

print(np.mean(q3_lst))
