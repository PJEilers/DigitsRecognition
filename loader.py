import numpy as np
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random as r
from PIL import Image, ImageOps

class Loader():

    train = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[],"9":[]}
    test = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}

    
    n_comp = 75;

    def __init__(self):
        self.data = np.genfromtxt("mfeat-pix.txt", dtype=None)
        for c in range(10):
            for idx in range(200):
                digit = np.reshape(self.data[ (200*c) + idx ], (16,15))
                if idx <= 99:
                    self.train[str(c)].append(digit)
                else:
                    self.test[str(c)].append(digit)

    def pca_init(self):
        self.pca_train = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
        self.pca_test = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}

    def pca(self, n_comp=75):
        self.pca_init()
        self.n_comp = n_comp;
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

    def getImage(self, c=-1, idx=-1, aug=False, set="train", flat=False):
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
            # print(im)
            im = im/6.0 #np.uint8(cm.binary(im) * 255)
            # print(np.uint8(im * 255))
            im = Image.fromarray(np.uint8(im*255), mode="L")

            im = ImageOps.pad(im, (31, 30), 3)#, (255, 255, 255, 255))
            angle = r.randint(-25, 25)
            im = im.rotate(angle)#, fillcolor=(255, 255, 255, 255))
            im = ImageOps.fit(im, (15, 16))

            im = np.array(im)

        if flat:
            im = im.flatten()

        return im

    def getNoisyImage(self, c=-1, idx=-1, aug=False, set="test", intensity = 0.1, flat=False):
        im = self.getImage(c=c, idx=idx, aug=False, set=set, flat=flat)
        #add Noise to image
        if flat:
            noise = np.random.rand(240)
        else:
            noise = np.random.rand(16,15)
        im = im + (intensity*noise)
        return im
    
    def getNoisyPcaImage(self, c=-1, idx=-1, set="test", intensity = 0.1):
        im = self.getPcaImage(c=c, idx=idx, set=set)
        noise = np.random.rand(self.n_comp)
        im = im + (intensity*noise)
        return im

    def getNoisySet(self, intensity=0.1, set="test", flat=False, shuffle=False, pca=False):
        x = []
        y = []
        if pca:
            for c in range(10):
                for idx in range(100):
                    x.append(self.getNoisyPcaImage(c=c, idx=idx, set=set, intensity=intensity))
                    y.append(c)            
        else:
            for c in range(10):
                for idx in range(100):
                    x.append(self.getNoisyImage(c=c, idx=idx, aug=False, set=set, intensity=intensity, flat=flat))
                    y.append(c)
            

        x = np.array(x)
        y = np.array(y)
        if shuffle:
            order = np.arange(x.shape[0])
            np.random.shuffle(order)
            x = x[order]
            y = y[order]

        return x,y

    def getWholeTrainSet(self, pca=False, shuffle=False, flat=False):
        x = []
        y = []

        dataset = None
        if pca==True:
            dataset=self.pca_train
        elif pca==False:
            dataset=self.train

        for c in range(10):
            for i in range(len(dataset[str(c)])):
                x.append(dataset[str(c)][i])
                y.append(c)

        x = np.array(x)
        y = np.array(y)
        
        if shuffle:
            order = np.arange(x.shape[0])
            np.random.shuffle(order)
            x = x[order]
            y = y[order]

        if flat:
            length = len(x)
                # x[i] = x[i].flatten()

            X_flatten = np.empty([length,16*15])
            for i in range (length):
                X_flatten[i] = x[i].flatten()
            x = X_flatten

        return x,y

    def getWholeTestSet(self, pca=False, shuffle=False, flat=False):
        x = []
        y = []

        dataset = None
        if pca == True:
            dataset = self.pca_test
        elif pca == False:
            dataset = self.test

        for c in range(10):
            for i in range(len(dataset[str(c)])):
                x.append(dataset[str(c)][i])
                y.append(c)

        x = np.array(x)
        y = np.array(y)
        if shuffle:
            order = np.arange(x.shape[0])
            np.random.shuffle(order)
            x = x[order]
            y = y[order]

        if flat:
            length = len(x)
                # x[i] = x[i].flatten()

            X_flatten = np.empty([length,16*15])
            for i in range (length):
                X_flatten[i] = x[i].flatten()
            x = X_flatten

        return x, y