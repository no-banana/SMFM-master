import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.callbacks.EarlyStopping
import re
import os
import requests
import pickle
from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def read_fasta(file_path):
    seq_list = []
    f = open(file_path,'r')
    for line in f:
        if '>' not in line:
            line = line.strip().upper()
            seq_list.append(line)
    return seq_list

def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

if __name__ == '__main__':
    file_path = './dataset/train.txt'
    label_path = './dataset/train_label.txt'
    sequences = read_fasta(file_path)
    indices = np.random.choice(len(sequences), len(sequences), replace=False)
    sequences = np.array(sequences)
    sequences = sequences[indices]

    label_ls = open(label_path).readlines()
    label = np.array(label_ls)
    label = to_categorical(label)
    label = label[:,-1]
    label = label[indices]

    sequences1 = sequences
    sequences = []
    Bert_Feature = []
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer(seq, 6)  # here 6 represents that we tokenize sequences using 6-mer
        sequences.append(ss)
    tokenizer = BertTokenizer.from_pretrained("The pre-trained model path", do_lower_case=False)
    model = TFAutoModelForSequenceClassification.from_pretrained("The pre-trained model path", num_labels=2, from_pt=True)
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    input_ids = np.array(torch.tensor(ids['input_ids']))
    token_type_ids = np.array(torch.tensor(ids['token_type_ids']))
    attention_mask = np.array(torch.tensor(ids['attention_mask']))

    id_train,id_val, y_train, y_val = train_test_split(input_ids, label, test_size=0.2, random_state=0)
    attention_train, attention_val, y_train, y_val = train_test_split(attention_mask, label, test_size=0.2, random_state=0)
    token_train, token_val, y_train, y_val = train_test_split(token_type_ids, label, test_size=0.2, random_state=0)

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy()
    )
    fit_callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, verbose=0, restore_best_weights=True)
        ]
    model.fit({'input_ids':id_train, 'attention_mask':attention_train, 'token_type_ids':token_train}, y=y_train, validation_data=({'input_ids':id_val, 'attention_mask':attention_val, 'token_type_ids':token_val}, y_val), epochs=5, verbose=1, batch_size=32, callbacks=fit_callbacks)
    model.save_pretrained("./fine-tune_models/enhancer_model_6mer")
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("./fine-tune_models/enhancer_model_6mer", from_tf=True)
    pytorch_model.save_pretrained("./fine-tune_models/pytorch_model_6mer")
