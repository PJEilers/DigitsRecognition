#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def SVM_noise():
    #in
    intensities = [.0, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50]
    dataset = Loader()
    dataset.pca(n_comp=75)

    #out
    result = []

    #code
    for i, noise in enumerate(intensities):
        accuracy = 0

        x_train, y_train = dataset.getNoisySet(intensity=noise, set="train", pca=True)
        x_test, y_test = dataset.getWholeTestSet(pca=True)
    
        for i in range(100):
            clf = svm.SVC(C=4.0)
            clf.fit(x_train, y_train)

            predict = clf.predict(x_test)
            matches = 0.0
            for j in range(len(predict)):
                if predict[j]==y_test[j]:
                    matches+=1

            accuracy += (matches/len(y_test))

        result.append(accuracy)

    return np.array(result)

