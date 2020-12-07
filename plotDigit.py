import numpy as np
from matplotlib import pyplot as plt

# load data
data = np.genfromtxt("mfeat-pix.txt", dtype=None)

# make a 2d array from a 1d array
first_digit = np.reshape(data[0], (16, 15))

# print values of first digit
print(first_digit)

# show first digit using the reversed gray map
plt.imshow(first_digit, cmap='gray_r', vmin=0, vmax=6)
plt.show()