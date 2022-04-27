import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
import pickle
from tqdm.auto import tqdm
import tensorflow as tf



def read_fasta(file_path):
    seq_list = []
    f = open(file_path,'r')
    for line in f:
        if '>' not in line:
            line = line.strip().upper()
            seq_list.append(line)
    return seq_list

def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def Enhancer_Bert(sequences, dataloader):
    features = []
    seq = []    
    tokenizer = BertTokenizer.from_pretrained("The path stores EnhancerBERT", do_lower_case=False)
    model = BertModel.from_pretrained("The path stores EnhancerBERT")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    for sequences in dataloader:
        seq.append(sequences)
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        embedding = embedding.last_hidden_state.cpu().numpy()
    
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd) 
    return features
    


def EnhancerBert(k):
    file_path = 'The path stores sequence file'
    sequences = read_fasta(file_path)
    sequences1 = sequences
    sequences = []
    Bert_Feature = []  
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer(seq, k)
        sequences.append(ss)
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=32, shuffle=False)
    Features = Enhancer_Bert(sequences, dataloader)
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature)
    bb = np.array(Bert_Feature)
    return bb

Embedding = EnhancerBert(3)
print(np.array(Embedding).shape)
np.save('./dynamic_semantic_information_3mer.npy',np.array(Embedding))
