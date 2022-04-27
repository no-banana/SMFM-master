import itertools
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn.ensemble import AdaBoostClassifier

DNAelements = 'ACGT'


elements = DNAelements

m2 = list(itertools.product(elements, repeat=2))
m3 = list(itertools.product(elements, repeat=3))
m4 = list(itertools.product(elements, repeat=4))
m5 = list(itertools.product(elements, repeat=5))

def readFASTAs(fileName):
    with open(fileName, 'r') as file:
        v = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                v.append(genome.upper())
                genome = ''
        v.append(genome.upper())
        del v[0]
        return v

def saveCSV(name, X, Y, type):
    if type == 'test':
        F = open((name + '_testDataset.csv'), 'w')

    else:
        if type == 'optimum':
            F = open((name + '_optimumDataset.csv'), 'w')

    for x, y in zip(X, Y):
        for each in x:
            F.write(str(each) + ',')
        F.write(str(int(y)) + '\n')
    F.close()

def readLabels(fileName):
    with open(fileName, 'r') as file:
        v = []
        for line in file:
            if line != '\n':
                v.append((line.replace('\n', '')).replace(' ', ''))
        return v

def kmers(seq, k):
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v

def pseudoKNC(x, k):
    t = []
    for i in range(1, k + 1, 1):
        v = list(itertools.product(elements, repeat=i))
        for i in v:
            t.append(x.count(''.join(i)))
    return t

def zCurve(x):
    t = []
    T = x.count('T'); A = x.count('A'); C = x.count('C'); G = x.count('G');
    x_ = (A + G) - (C + T)
    y_ = (A + C) - (G + T)
    z_ = (A + T) - (C + G)
    t.append(x_); t.append(y_); t.append(z_)
    return t

def gcContent(x):
    t = []
    T = x.count('T')
    A = x.count('A');
    C = x.count('C');
    G = x.count('G');
    t.append( (G + C) / (A + C + G + T)  * 100.0 )
    return t


def cumulativeSkew(x):
    
    t = []
    T = x.count('T')
    A = x.count('A');
    C = x.count('C');
    G = x.count('G');

    GCSkew = (G-C)/(G+C)
    ATSkew = (A-T)/(A+T)

    t.append(GCSkew)
    t.append(ATSkew)
    return t


def atgcRatio(x):
    t = []
    T = x.count('T')
    A = x.count('A');
    C = x.count('C');
    G = x.count('G');

    t.append( (A+T)/(G+C) )
    return t


def monoMonoKGap(x, g):  # k=1, m=1
    m = m2
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            t.append(C)
    return t

def monoDiKGap(x, g):  # k=1, m=2
    t = []
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C)
    return t

def diMonoKGap(x, g):  # k=2, m=1
    t = []
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C)
    return t
def monoTriKGap(x, g):  # k=1, m=3
    t = []
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-3] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            t.append(C)
    return t
def triMonoKGap(x, g):  # k=3, m=1
    t = []
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            t.append(C)
    return t
def diDiKGap(x, g): # k=2, m=2
    m = m4
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            t.append(C)
    return t
def diTriKGap(x, g):  # k=2, m=3
    m = m5
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 5)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-3] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            t.append(C)
    return t
def triDiKGap(x, g):  # k=3, m=2
    m = m5
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 5)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            t.append(C)
    return t
def generateFeatures(kGap, kTuple, x, y):
        feature = []
        t = zCurve(x)         
        for item in t:
            feature.append(item)
        t= gcContent(x)           
        for item in t:
            feature.append(item)
        t = cumulativeSkew(x)      
        for item in t:
            feature.append(item)
        t = atgcRatio(x)         
        for item in t:
            feature.append(item)
        t = pseudoKNC(x, kTuple)            
        for item in t:
            feature.append(item)
        t = monoMonoKGap(x, kGap)      
        for item in t:
            feature.append(item)
        t = monoDiKGap(x, kGap)        
        for item in t:
            feature.append(item)
        t = monoTriKGap(x, kGap)       
        for item in t:
            feature.append(item)
        t = diMonoKGap(x, kGap)        
        for item in t:
            feature.append(item)
        t = diDiKGap(x, kGap)          
        for item in t:
            feature.append(item)
        t = diTriKGap(x, kGap)        
        for item in t:
            feature.append(item)
        t = triMonoKGap(x, kGap)       
        for item in t:
            feature.append(item)
        t = triDiKGap(x, kGap)         
        for item in t:
            feature.append(item)
        feature.append(y)
        return feature

def main(args):
    # here have to give the definitions of X, Y
    X = readFASTAs(args.sequences)
    print('loading sequence complete')
    Y = readLabels(args.labels)
    print('loading label complete')
    Y = LabelEncoder().fit_transform(Y)
    
    # The number of X and Y must be equal. 
    assert len(X)==len(Y)
    T = []
    print('start generating features')
    for x, y in zip(X, Y):
        feature = generateFeatures(5, 3, x, y)
        print(len(feature))
        T.append(feature)
        print(len(T))
    T = np.array(T)

    X = T[:,:-1]
    Y = T[:,-1]
    print('Finish generating features')

    if args.testDataset == 1:
        print('Saving the test .csv file')
        saveCSV(args.name, X, Y, 'test')
        print('finish')
        return
    else:
        print('Start selecting best feature subset')
        model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=500, learning_rate=1.0)
        model.fit(X, Y)
        importantFeatures = model.feature_importances_
        feature_values = np.sort(importantFeatures)[::-1]
        num = importantFeatures.argsort()[::-1][:len(feature_values[feature_values>0.00])]
        F = open((args.name + '_selectedIndex.txt'), 'w')
        ensure = True
        for i in num:
            if ensure:
                F.write(str(i))
            else:
                F.write(','+str(i))
            ensure = False
        F.close()
        X = X[:, num]
        saveCSV(args.name, X, Y, 'optimum')
        print('finish')
        return




if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sequences', type=str, help='the file stores sequences')
    p.add_argument('--labels', type=str, help='the file stores labels corresponding sequences')
    p.add_argument('--testDataset', type=int, help='whether the set that will be generated is a test set', default=0, choices=[0, 1])
    p.add_argument('--name', type=str, help='the dataset name', default='')
    args = p.parse_args()
    main(args)

