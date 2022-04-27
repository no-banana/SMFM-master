from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import deepforest as DF
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def readSeq(seq_path, label_path):
    file = open(seq_path)
    ls = file.readlines()
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
    return seq_ls, np.array(label_ls)



if __name__ == '__main__':
    file_path = './multi-source_biological_features/train_optimumDataset.csv'
    ind_file_path = './multi-source_biological_features/test_ind_fullDataset.csv'
    selected_index_path = './multi-source_biological_features/train_selectedIndex.txt'

    bert_path = './dynamic_semantic_information/train_contextual_inforamtion.npy'
    ind_bert_path = './dynamic_semantic_information/test_contextual_inforamtion.npy'
    np.random.seed(46)
    # train dataset
    w_vector = np.load(bert_path)
    print(w_vector.shape)
    feature = pd.read_csv(file_path, header=None)
    print(feature)
    X = feature.iloc[:, :-1].values
    Y = feature.iloc[:, -1].values
    scale = StandardScaler()
    X = scale.fit_transform(X)
    biological_X = X
    X = np.concatenate((X, w_vector), axis=1)
    indexes = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[indexes]
    Y = Y[indexes]
    print(X.shape)
    print(Y.shape)

    # test dataset
    ind_w_vector = np.load(ind_bert_path)
    print(ind_w_vector.shape)
    ind_feature = pd.read_csv(ind_file_path, header=None)
    print(ind_feature)
    ind_X = ind_feature.iloc[:, :-1].values
    ind_Y = ind_feature.iloc[:, -1].values
    F = open(selected_index_path, 'r')
    v = F.read().split(',')
    v = [int(i) for i in v]
    ind_X = ind_X[:, v]

    scale = StandardScaler()
    ind_X = scale.fit_transform(ind_X)
    ind_X = np.concatenate((ind_X, ind_w_vector), axis=1)
    ind_indexes = np.random.choice(ind_X.shape[0], ind_X.shape[0], replace=False)
    ind_X = ind_X[ind_indexes]
    ind_Y = ind_Y[ind_indexes]
    print(ind_X.shape)
    print(ind_Y.shape)

    model1 = DF.CascadeForestClassifier(random_state=21, n_estimators=65, use_predictor=True, predictor='lightgbm',
                                        delta=5e-6, max_layers=30, n_trees=25)
    model1.fit(X, Y)
    pre_label1 = model1.predict(ind_X)
    print(accuracy_score(y_true=ind_Y, y_pred=pre_label1))
    model2 = SVC(random_state=0, C=5.0, kernel='poly', gamma=0.001)
    model2.fit(X, Y)
    pre_label2 = model2.predict(ind_X)
    print(accuracy_score(y_true=ind_Y, y_pred=pre_label2))
    model3 = RandomForestClassifier(random_state=94, n_estimators=75, criterion='gini')
    model3.fit(X, Y)
    pre_label3 = model3.predict(ind_X)
    print(accuracy_score(y_true=ind_Y, y_pred=pre_label3))

    ensemble = pre_label1 + pre_label2 + pre_label3
    pre_label = []
    for item in ensemble:
        if item > 1:
            pre_label.append(1)
        else:
            pre_label.append(0)

    pre_label = np.array(pre_label)

    acc = accuracy_score(y_true=ind_Y, y_pred=pre_label)
    mcc = matthews_corrcoef(y_true=ind_Y, y_pred=pre_label)
    sn = recall_score(y_true=ind_Y, y_pred=pre_label)
    sp = (acc * len(ind_Y) - sn * sum(ind_Y)) / (len(ind_Y) - sum(ind_Y))
    print('acc: ' + str(acc))
    print('mcc: ' + str(mcc))
    print('sn: ' + str(sn))
    print('sp: ' + str(sp))


