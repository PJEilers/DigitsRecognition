import os
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random as r
from PIL import Image, ImageOps

class Loader():

    train = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[],"9":[]}
    test = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}

    pca_train = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
    pca_test = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}

    def __init__(self):
        self.data = np.genfromtxt("mfeat-pix.txt", dtype=None)
        for c in range(10):
            for idx in range(200):
                digit = np.reshape(self.data[ (200*c) + idx ], (16,15))
                if idx <= 99:
                    self.train[str(c)].append(digit)
                else:
                    self.test[str(c)].append(digit)


    def pca(self, n_comp=75):
        scaled = StandardScaler().fit_transform(self.data)
        pca_digit = PCA(n_components=n_comp)
        pca_done = pca_digit.fit_transform(scaled)
        for c in range(10):
            for idx in range(200):
                digit = pca_done[ (200*c) + idx]
                if idx<= 99:
                    self.pca_train[str(c)].append(digit)
                else:
                    self.pca_test[str(c)].append(digit)

    def getPcaImage(self, c=-1, idx=-1, set="train"):
        #check if pca has been done
        #if not, do it
        if len(self.pca_train["0"])==0:
            self.pca()
        # First pick a class and index if none are provided
        if c == -1:
            c = r.randint(0, 9)
        if idx == -1:
            idx = r.randint(0, 99)
        # Pick from either the train or test set
        if set == "train":
            im = self.pca_train[str(c)][idx]
        else:
            im = self.pca_test[str(c)][idx]

        return im

    def getImage(self, c=-1, idx=-1, aug=False, set="train"):
        #First pick a class and index if none are provided
        if c == -1:
            c = r.randint(0,9)
        if idx == -1:
            idx = r.randint(0,99)
        #Pick from either the train or test set
        if set=="train":
            im = self.train[str(c)][idx]
        else:
            im = self.test[str(c)][idx]
        #Apply data augmentation if desired
        if aug==True:
            im = im/6.0
            im = Image.fromarray(np.uint8(cm.binary(im) * 255))
            im = ImageOps.pad(im, (31, 30), 3, (255, 255, 255, 255))
            angle = r.randint(-25, 25)
            im = im.rotate(angle, fillcolor=(255, 255, 255, 255))
            im = ImageOps.fit(im, (16, 15))
            im = np.array(im)

        return im

    def getWholeTrainSet(self, pca=False):
        x = []
        y = []

        dataset = None
        if pca==True:
            dataset=self.pca_train
        elif pca==False:
            dataset=self.train

        for c in range(10):
            for i in range(len(self.dataset[str(c)])):
                x.append(self.dataset[str(c)][i])
                y.append(c)

        return x,y

    def getWholeTestSet(self, pca=False):
        x = []
        y = []

        dataset = None
        if pca == True:
            dataset = self.pca_test
        elif pca == False:
            dataset = self.test

        for c in range(10):
            for i in range(len(self.dataset[str(c)])):
                x.append(self.dataset[str(c)][i])
                y.append(c)

        return x, y