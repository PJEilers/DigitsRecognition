# DigitsRecognition
Using machine learning to recognize handwritten digits

## How to use the Loader - by Alessandro

It's important to put the loader.py file in the same folder where the text file containing the dataset is.
It can be imported by doing `from loader import Loader`

It is initialized by calling the constructore, i.e. `dataset = Loader()`
Through the dataset variable we can access both normal images (16x15) and feature vectors extracted throgh PCA.

To perform PCA we can call the function `dataset.pca(n_comp=75)`. 75 Components account for slightly over than 93% of the variance, you can try with less or more variance.

To get an image we can call the `dataset.getImage()` function. This function has 4 optional attributes we can specify:
1. **c** which stands for the class from which we select an image. It can go from 0 to 9, both included.
1. **idx** which stands for the index of the image. It can go from 0 to 99, both included.
1. **aug** which declares if you want to apply data augmentation which consists of a random rotation between -25 and 25 degrees. It can be `True` or `False`.
1. **set** which states if we want to select an image from the train set or the test set. It can either be `"train"` or `"test"`.

We can also select a PCAed feature vector by using the analogous function `dataset.getPcaImage()` with the same arguments as before. 
These functions return a single variable which is the exctracted image.

We can also extract the whole dataset, split into train and set using two functions, `dataset.getWholeTrainSet()` and `dataset.getWholeTestSet()`. These function only have one optional argument which is **pca** that can be either `True` or `False` and specifies if we want to select the normal images or the PCA feature vectors. These last two funtion returns a tuple `(x,y)` where x is a list of examples and y is a list of labels.

### Noise update

To get an image with some noise added, we can use the `dataset.getNoisyImage()` function. The function accept a new parameter called **intensity** to decide the intensity of the noise added (default is `0.1`). On top of this, the function also retains the already introduced parameters **c**, **idx**, **aug** and **set** as already explained before.

To get the whole test set with noise added, we can use the `dataset.getNoisySet()` function. Intensity can also be set with the same **intensity** argument for this function. Furthermore, if we want to obtain a noisy train set for doing experiment, with this functuion we can also use the `set="train"` additional parameter.

