from matrix_image import ImageHandler
import numpy as np 

ih = ImageHandler("WaterfallWithText.png")
A = ih.compose_grayscale_mat()
print A[0:10]
print "\n"

M, N = A.shape
patch_size = 3
num_patches = (N-(patch_size-1)) * (M - (patch_size - 1))
x = np.zeros((num_patches, patch_size**2))

patch_id = 0

for i in range(M - (patch_size - 1)):
	for j in range(N-(patch_size - 1)):
		x[patch_id] = (A[i:i+patch_size:1, j:j+patch_size:1]).flatten()
		patch_id += 1

print x[0:10]