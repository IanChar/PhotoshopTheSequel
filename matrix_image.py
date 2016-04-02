from scipy import misc
import matplotlib.pyplot as plt

im = misc.face()
im = misc.imread('ian.jpeg') # imports image as a numpy array

plt.imshow(im)
plt.show()