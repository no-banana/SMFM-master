import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Convolution1D, Dense
from sklearn.model_selection import train_test_split
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.7
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate


def readSeq(label_path):
    ls = []
    seq_ls = []
    for item in ls:
        if item[0] != '>':
            seq_ls.append(item.strip())
    print(len(seq_ls))
    file = open(label_path)
    ls = file.readlines()
    label_ls = []
    for item in ls:
        label_ls.append(item.strip())
    print(len(label_ls))
    return np.array(label_ls)


if __name__ == '__main__':
    trainlabel_path = './dataset/train_label.txt'
    testlabel_path = './test_label.txt'
    bert_path = './original_information/train.npy'
    ind_bert_path = './original_information/test.npy'

    # original dynamic semantic information extracted from EnhancerBERT of training set
    train_bert = np.load(bert_path)
    train_seq, train_label = readSeq(trainlabel_path)
    train_label = to_categorical(train_label)

    # original dynamic semantic information extracted from EnhancerBERT of independent test set
    test_bert = np.load(ind_bert_path)
    test_seq, test_label = readSeq(testlabel_path)
    test_label = to_categorical(test_label)
    w2v_input = Input(shape=(train_bert.shape[1], train_bert.shape[2]), name='w2v_input')
    w2v = Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu')(w2v_input)
    w2v = Convolution1D(filters=16, kernel_size=3, padding='same', activation='relu', name='CNN_output')(w2v)
    w2v = Bidirectional(LSTM(units=16, return_sequences=False), name='before_Dense')(w2v)
    w2v = Dense(units=16, activation='relu')(w2v)
    w2v_output = Dense(units=2, name='w2v_output', activation='softmax')(w2v)
    w2v_model = Model(inputs=w2v_input, outputs=w2v_output)
    w2v_model.summary()
    w2v_model.compile(optimizer='nadam', loss={'w2v_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min', restore_best_weights=True),
                 LearningRateScheduler(step_decay)]

    train_X, val_X, train_Y, val_Y = train_test_split(train_bert, train_label, test_size=0.135, stratify=train_label)

    # train the network using splited training set and validation set
    w2v_model.fit(x=train_X, y=train_Y, epochs=30, batch_size=16, verbose=1, shuffle=True, callbacks=callbacks,
                  validation_data=({'w2v_input': val_X}, {'w2v_output': val_Y}))
    model.save('./SMFM_dl_netowrk.h5')