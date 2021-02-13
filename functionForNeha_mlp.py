from sklearn.linear_model import SGDClassifier
import numpy as np
from matplotlib import pyplot as plt
from loader import Loader



def LogRegression():
    #in
    intensities = [.0, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50]
    dataset = Loader()

    #out
    result = []

    #code
    for i, noise in enumerate(intensities):
        partial_result = []

        x_train, y_train = dataset.getNoisySet(intensity=noise, set="train", flat=True)
        x_test, y_test = dataset.getWholeTestSet(flat=True)

        for i in range(100):
            clf = neural_network.MLPClassifier(hidden_layer_sizes=(250, 100)) 
            clf.fit(x_train, y_train)

            predict = clf.predict(x_test)
            matches = 0.0
            for j in range(len(predict)):
                if predict[j]==y_test[j]:
                    matches+=1

            accuracy = matches/len(predict)
            partial_result.append(accuracy)

        avg = np.mean(partial_result)
        result.append(avg)

    return np.array(result)

if __name__=="__main__":
    v = LogRegression()
    print(v)