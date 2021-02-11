from sklearn.linear_model import SGDClassifier
import numpy as np
from matplotlib import pyplot as plt
from loader import Loader

def evaluate(received, ppredict):
    mismatch = 0
    for i in range(len(ppredict)):
        if received[i] != ppredict[i]:
            mismatch += 1

    return mismatch/len(received)

def flatten_image (X, length):
    X_flatten = np.empty([length,16*15])
    for i in range (length):
        X_flatten[i] = X[i].flatten()
    return X_flatten

final = []
classes=np.array([0,1,2,3,4,5,6,7,8,9])

dataset = Loader()
#dataset.pca(71)

#x_train, y_train = dataset.getWholeTrainSet(shuffle=True, pca=False)
x_train, y_train = dataset.getNoisySet(intensity=0.25, set="train", flat=True)
#x_train = flatten_image(x_train, len(y_train))
x_test, y_test = dataset.getWholeTrainSet(flat=True)
#x_test, y_test = dataset.getNoisySet(intensity=0.5, set="test", flat=True)
#x_test = flatten_image(x_test, len(y_test))

x = [ x_train[:200], x_train[200:400], x_train[400:600], x_train[600:800], x_train[800:1000] ]
y = [ y_train[:200], y_train[200:400], y_train[400:600], y_train[600:800], y_train[800:1000] ]

xx =  [ x_test[:200], x_test[200:400], x_test[400:600], x_test[600:800], x_test[800:1000] ]
yy = [ y_test[:200], y_test[200:400], y_test[400:600], y_test[600:800], y_test[800:1000] ]

clf = SGDClassifier(loss="log", max_iter=2000, tol=1e-3)

for i in range(len(x)):
    xtest = None
    xtrain = np.array([])
    ytest = None
    ytrain = np.array([])
    for j in range(len(x)):
        if j==i:
            xtest = xx[i]
            ytest = yy[i]
        else:
            #xtrain = np.concatenate((xtrain, np.array(x[j])))
            #ytrain = np.concatenate((ytrain, np.array(y[j])))
            clf.partial_fit(x[j], y[j], classes=classes)





    #clf.fit(xtrain, y_train)

    predict = clf.predict(xtest)

    test_err = evaluate(ytest, predict)
    final.append(test_err)

print((1-np.array(final))*100, 100*(1-np.mean(final)))
#print(final, np.mean(final))
