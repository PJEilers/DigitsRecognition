from sklearn.linear_model import SGDClassifier
import numpy as np
from matplotlib import pyplot as plt
from loader import Loader
import time

def evaluate(y, predict):
    mismatch = 0
    for i in range(len(predict)):
        if y[i] != predict[i]:
            mismatch += 1

    return mismatch/len(y)

def flatten_image (X, length):
    X_flatten = np.empty([length,16*15])
    for i in range (length):
        X_flatten[i] = X[i].flatten()
    return X_flatten


tr_err = []
te_err = []

dataset = Loader()
dataset.pca(n_comp=71)

#x_train, y_train = dataset.getWholeTrainSet(shuffle=True, pca=True)
x_train, y_train = dataset.getNoisySet(intensity=0.5, set="train", flat=True)
#x_train = flatten_image(x_train, len(y_train))

x_test, y_test = dataset.getNoisySet(intensity=.5, flat=True)
#x_test, y_test = dataset.getWholeTestSet(pca=False)
#x_test = flatten_image(x_test, len(y_test))

for i in range(100):
    print(i, end="\r")
    clf = SGDClassifier(loss="log", max_iter=2000, tol=1e-3)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    train_pred = clf.predict(x_train)

    test_error = evaluate(y_test, predict)
    train_error = evaluate(y_train, train_pred)

    tr_err.append(train_error)
    te_err.append(test_error)





print("Train", 100*np.mean(tr_err), "Test", 100*np.mean(te_err))





