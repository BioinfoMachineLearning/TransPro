#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import os,sys
import argparse
from Bio import SeqIO
import itertools
import string

from data import *
from modules import *

import numpy as np

from itertools import islice

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

def read_label(filename: str):
    label = np.load(filename)
    return label

def rebatch(b,a3m_file,pad_idx = 1,max_positions=500):
    train_msa = []
    tor_label = []
    ss_label = []
    for t in list(b):
        if not os.path.exists(a3m_file):
            continue
        train_msa.append(read_msa(a3m_file,max_positions))
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

def get_lst(src_dir, a3m_file, lst):
    base=os.path.basename(a3m_file)
    tar = os.path.splitext(base)[0]
    lst = src_dir+"/"+lst
    f = open(lst, "w")

    with open(a3m_file) as myfile:
        head = list(islice(myfile, 2))
    seq = head[-1].rstrip()
    f.write(tar+" "+str(len(seq)))
    f.close()
    return lst



if __name__=="__main__":
    #### command line argument parsing ####
    parser = argparse.ArgumentParser()
    parser.description="TransPross:1D transformer for predicting protein secondary structure prediction"
    parser.add_argument("-i", "--input", help="MSA in the format a3m", type=is_file, required=True)

    args = parser.parse_args()

    #User setting parameters
    a3m_file = os.path.abspath(args.input)

    # GPUs to use
    if torch.cuda.is_available():
        cur_id = "cuda:" + str(torch.cuda.current_device())
        devices = [torch.cuda.current_device()]
        print("You're using "+cur_id+"....")
        print('Predicting......')
    else:
        cur_id = "cpu"
        print("You're using "+cur_id+"....")
        print("This may take longer....Please wait.....")
        devices = [-1]

    # TransPross parameters
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

    test_lst = get_lst(src_dir, a3m_file, "test.lst")

    ############################Prediction##########################
    q3_lst = []
    model_dir = ["model/model_current_a3m3","model/model_current_a3m4","model/model_current_a3m1","model/model_current_a3m9","model/model_current_a3m11"]

    test_iter = MyIterator(batch_size=1, L_lst=test_lst ,train=False)

    for i, batch_orig in enumerate(test_iter.batches):
        out = []
        start = time.time()
        tar = list(batch_orig.keys())[0] 

        batch = rebatch(batch_orig, a3m_file, pad_idx,max_positions)
        src_mask = batch.src_mask
        src = batch.src
        src = batch.src.to(device=cur_id)
        batch_size, n, seq_l = src.size()
        src_mask = src_mask.to(device=cur_id)
        
        for model_file in model_dir:
            if os.path.exists(model_file):
                if torch.cuda.is_available():
                    model = torch.load(model_file)
                else:
                    model = torch.load(model_file, map_location=torch.device('cpu'))
            else:
                print(model_file+" doesn't exist, please download from https://doi.org/10.5281/zenodo.6762376")
                sys.exit()
            
            model_par = nn.DataParallel(model, device_ids=devices)
            #model_par = model
            
            model_par.eval()
            out3,out3_prob = greedy_decode2(model, src, src_mask, max_len=seq_l, start_symbol=0)
        
            out.append(out3_prob)
            
        a = np.array(out)
        out = np.average(a, axis=0)
        b = np.argmax(out, axis=1)
        out3_lst = b.tolist()

        print("Prediction for target "+tar+":", end="\t")
        sym = ""
        pred = ""
        pred_lst = []
        sym_lst = []
        for i in range(len(out3_lst)):
            sym += TGT_itos_3(out3_lst[i])
            if sym == "</s>": break
            pred_lst.append(out3_lst[i])
        print(sym)

    os.system("rm "+src_dir+"/test.lst")
    print("Done")