import numpy as np
from matplotlib import pyplot as plt
from loader import Loader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn import svm

dataset = Loader()

train = []
test = []

for i in range(240):
    print("Doing components ", str(i+1))
    dataset = Loader()
    dataset.pca(i+1)
    x_train, y_train = dataset.getWholeTrainSet(pca=True, shuffle=True)
    x_test, y_test = dataset.getWholeTestSet(pca=True, shuffle=True)

    print(x_train.shape, x_test.shape)

    #clf = svm.SVC()
    clf = SGDClassifier(loss="squared_hinge" ,max_iter=1000, tol=1e-3)
    clf.fit(x_train, y_train)

    mismatch=0
    y_predict=clf.predict(x_test)
    for i in range(len(y_predict)):
        if y_test[i] != y_predict[i]:
            mismatch+=1

    test.append(mismatch/len(y_test))

    mismatch = 0
    y_predict = clf.predict(x_train)
    for i in range(len(y_predict)):
        if y_train[i] != y_predict[i]:
            mismatch += 1

    train.append(mismatch/len(y_train))


fig, ax = plt.subplots()
x = np.arange(1,241,1)
ax.plot(x, test, color="red", label="Test error")
ax.plot(x, train, color="blue", label="Train error")

min_val = min(test)
index = test.index(min_val)

ax.plot([x[index], x[index]], [0, max(test)], color="gray")

ax.set_xlabel("Number of components considered")
ax.set_ylabel("Error")
ax.legend()

plt.show()