#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
from Bio import SeqIO
import itertools
import string

from data import *
from modules import *

import numpy as np
import sys

from sklearn.metrics import mean_absolute_error


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

def rebatch(b,pad_idx = 1,max_positions=500):
    train_msa = []
    tor_label = []
    ss_label = []
    for t in list(b):
        if not os.path.exists(train_dir+"/a3m/"+t+".a3m"):
            continue
        if not os.path.exists(train_dir+"/phi_psi/"+t+".npy"):
            continue
        if not os.path.exists(train_dir+"/ss/ss_3/"+t+".npy"):
            continue
        train_msa.append(read_msa(train_dir+"/a3m/"+t+".a3m",max_positions))
        tor_label.append(read_label(train_dir+"/phi_psi/"+t+".npy"))
        ss_label.append(read_label(train_dir+"/ss/ss_3/"+t+".npy"))
    #- . <cls> <eos> <mask> <null_1> <pad> <unk> A B C D E F G H I K L M N O P Q R S T U V W X Y Z
    #30 29 0 2 32 31 1 3 5 25 23 13 9 18 6 21 12 15 4 20 17 28 14 16 10 8 11 26 7 22 24 19 27
  
    msa_alphabet = Alphabet.from_architecture()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    msa_batch_labels_de1,msa_batch_labels_de2,msa_batch_labels_de3,msa_batch_strs, msa_batch_tokens = msa_batch_converter(train_msa,tor_label,ss_label)
    return Batch(msa_batch_tokens, msa_batch_labels_de1,msa_batch_labels_de2,msa_batch_labels_de3,pad_idx)

def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname

def is_file(filename):
    """Checks if a file is an invalid file"""
    if not os.path.exists(filename):
        msg = "{0} doesn't exist".format(filename)
        raise argparse.ArgumentTypeError(msg)
    else:
        return filename

def mkdir_if_not_exist(tmpdir):
    ''' create folder if not exists '''
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)


if __name__=="__main__":
    #### command line argument parsing ####
    parser = argparse.ArgumentParser()
    parser.description="TransPross:1D transformer for predicting protein secondary structure prediction"
    parser.add_argument("--data_dir", help="folder path for storing data", type=is_dir, required=True)
    parser.add_argument("--dataset", help="train", type=str)
    parser.add_argument("--model_num", help="training list id",default=1, type=int)
    parser.add_argument("--N", help="number of attention layers",default=6, type=int)
    parser.add_argument("--max_positions", help="maximum number of sequences allowed in the input MSA",default=1500, type=int)
    parser.add_argument("--BATCH_SIZE", help="batch size",default=5, type=int)

    args = parser.parse_args()
    input = os.path.abspath(args.indir)
    #target = args.target

    #User setting parameters
    model_num = args.model_num
    N = args.N
    max_positions = args.max_positions
    BATCH_SIZE = args.BATCH_SIZE
    data_dir = args.data_dir
    dataset = args.dataset
    
    # GPUs to use
    devices = [0]
    pad_idx = 1
    padding_idx=1
    SRC_vocab = 33
    TGT_vocab = 5
    d_model=512
    d_ff=2048
    h=8
    dropout=0.1

    src_dir = os.path.dirname(os.path.abspath(__file__))

    model = make_model(SRC_vocab, TGT_vocab, N, d_model, d_ff, h, dropout, max_positions, pad_idx)
    model.cuda()

    criterion1 = MSE(padding_idx=padding_idx)
    criterion1.cuda()

    criterion2 = LabelSmoothing(size=TGT_vocab, padding_idx=padding_idx, smoothing=0.1)
    criterion2.cuda()

    train_dir = os.path.join(data_dir, dataset)
    train_lst = os.path.join(train_dir, "lst/model"+model_num+"/train"+model_num+".lst")
    val_lst = os.path.join(train_dir, "lst/model"+model_num+"/valid"+model_num+".lst")
    test_lst = os.path.join(train_dir, "test.lst")

    model_file = os.path.join(src_dir,"model/model_best_a3m"+model_num)
    if os.path.exists(model_file):
        model = torch.load(model_file)

    model_par = nn.DataParallel(model, device_ids=devices)
    #model_par = model
    model_opt = NoamOpt(d_model, 1, 4000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    avg_mae_lst = []
    for epoch in range(200):
        print("Running epoch:{}".format(epoch))
        model_par.train()
        train_iter = MyIterator(batch_size=BATCH_SIZE, L_lst=train_lst ,train=True)
        valid_iter = MyIterator(batch_size=10, L_lst=val_lst ,train=False)
        loss = run_epoch((rebatch(b, pad_idx,max_positions) for b in train_iter.batches), model_par,  SimpleLossCompute(model.generator1,model.generator2, criterion1,criterion2,model_opt))
        print("train loss:{}".format(loss))
        model_par.eval()
        # loss = run_epoch((rebatch(b, pad_idx, max_positions) for b in valid_iter.batches), model_par, SimpleLossCompute(model.generator, criterion, None))
        # print("valid loss:{}".format(loss))
        # print("valid acc:{}".format(acc))
        torch.save(model,src_dir+"/model/model_current_a3m"+model_num)
        ############################Prediction##########################
        if epoch % 50 == 1:
            phi_mae_lst = []
            psi_mae_lst = []
            acc_lst = []
            test_iter = MyIterator(batch_size=1, L_lst=val_lst ,train=False)
            
            for i, batch in enumerate(test_iter.batches):
                start = time.time()
                tar = list(batch.keys())[0] 
                if not os.path.exists(train_dir+"/a3m/"+tar+".a3m"):
                    #print(train_dir+"/aln/a3m/"+tar+".a3m doesn't exist....")
                    continue
                if not os.path.exists(train_dir+"/phi_psi/"+tar+".npy"):
                    #print(train_dir+"/aln/a3m/"+tar+".a3m doesn't exist....")
                    continue
                else:
                    batch = rebatch(batch, pad_idx,max_positions)
                    src_mask = batch.src_mask
                    src = batch.src
                    src = batch.src.to(device="cuda:0")
                    batch_size, n, seq_l = src.size()
                    src_mask = src_mask.to(device="cuda:0")
                    out1,out2,out3 = greedy_decode(model, src, src_mask, max_len=seq_l, start_symbol=0)
                    out1_lst = out1[0,0:-1].cpu().tolist()
                    out2_lst = out2[0,1:].cpu().tolist()
                    out2_lst[-1] = 0.0
                    trg_y1 = batch.trg1_y
                    trg_y1_lst = trg_y1[0].tolist()
                    trg_y2 = batch.trg2_y
                    trg_y2_lst = trg_y2[0].tolist()
                    out1_lst = [x * 180 for x in out1_lst]
                    out2_lst = [x * 180 for x in out2_lst]
                    trg_y1_lst = [x * 180 for x in trg_y1_lst]
                    trg_y2_lst = [x * 180 for x in trg_y2_lst]
                    
                    out3_lst = out3[0,1:].cpu().tolist()
                    trg_y3 = batch.trg3_y
                    trg_y3_lst = trg_y3[0].tolist()
                    acc_lst.append(accuracy_score(trg_y3_lst, out3_lst))
                    
                    mae_phi = mean_absolute_error(trg_y1_lst, out1_lst)
                    phi_mae_lst.append(mae_phi)
                    
                    mae_psi = mean_absolute_error(trg_y2_lst, out2_lst)
                    psi_mae_lst.append(mae_psi)
                                  
                    if i % 50 == 1:
                        elapsed = time.time() - start
                        print("Epoch Step: %d Sec: %f" %(i, elapsed))
                        start = time.time()
                    
            print(np.mean(phi_mae_lst))
            print(np.mean(psi_mae_lst))
            print(np.mean(acc_lst))
            mse_avg = (np.mean(phi_mae_lst) + np.mean(psi_mae_lst))/2
            if (not avg_mae_lst) or (mse_avg < min(avg_mae_lst)):
                avg_mae_lst.append(mse_avg)
                torch.save(model,src_dir+"/model/model_best_a3m"+model_num)
