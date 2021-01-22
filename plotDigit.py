# This file is used to take a sample from the dataset to be shown in the report.

# import libraries
import numpy as np
from matplotlib import pyplot as plt
import random

# load data
data = np.genfromtxt("mfeat-pix.txt", dtype=None)

# set parameters
n_cols = 3
n_rows = 5
indices = random.sample(range(200), n_rows)

# initialize subplots
fig, axes = plt.subplots(n_cols, n_rows, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

# fill the figure
for x in range(n_cols):
    for y in range(n_rows):
        # make a 2d array from a 1d array
        index = indices[y] + x * 800 + 200
        first_digit = np.reshape(data[index], (16, 15))

        # show image
        axes[x,y].imshow(first_digit, cmap='gray_r')

        # remove ticks
        axes[x,y].set_xticks([])
        axes[x,y].set_yticks([])

# show figure
plt.show()