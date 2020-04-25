


import numpy as np


def labelEncode(Y):
    
    labels = np.unique(Y)
    encoder = dict()
    
    for i in range(len(labels)):
        encoder[labels[i]] = i+1
    
    encoded = []
    
    for i,label in enumerate(Y):
        encoded.append(encoder[label])
        
    return np.array(encoded)

def test_train_split(X, y,p = 0.6):

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    slice_portion = int(p*len(y))

    train_slice, test_slice = indices[:slice_portion], indices[slice_portion+1:]
    X_train, X_test, y_train, y_test = X[train_slice], X[test_slice], y[train_slice], y[test_slice]
    # print(X_test,y_test)
    return X_train, X_test, y_train, y_test