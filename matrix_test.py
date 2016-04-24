from matrix_image import ImageHandler
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

patch_size = 99


def disp_greyscale(A):
	ih.set_color_matrix('r', A)
	ih.set_color_matrix('g', A)
	ih.set_color_matrix('b', A)
	ih.render_img_mat()



def encode_image(A):

	M, N = A.shape
	num_patches = (N-(patch_size-1)) * (M - (patch_size - 1))
	X_t = np.zeros((num_patches, patch_size**2))

	patch_id = 0

	for i in range(M - (patch_size - 1)):
		for j in range(N-(patch_size - 1)):
			X_t[patch_id] = (A[i:i+patch_size:1, j:j+patch_size:1]).flatten()
			patch_id += 1

	return X_t, M, N

def count_patch_coord(patch_id, M, N, patch_size):
	'''	count the coordinate (i,j) that the first element of
	patch is pointing to in the orignal image

	Input:
		patch_id - number of patch in X'
		M,N - dimension of our image MxN
	'''
	i = int(patch_id/(M - (patch_size - 1))) # get row number, ex: 0'th row is any patch < M - (n-1)
	j = patch_id % (M - (patch_size - 1)) # get column number
	return i,j

def put_patch_back(base_i, base_j, row, patch_size, A_new, pixel_freq_map):
	for z in range(len(row)):
		i = base_i + int(z/patch_size)
		j = base_j + (z % patch_size)
		try:
			A_new[i][j] += row[z]
			if (i,j) in pixel_freq_map:
				pixel_freq_map[(i,j)] += 1
			else:
				pixel_freq_map[(i,j)] = 1
		except:
			pass

def decode_image(X_t, M, N):
	#patch_size = self.patch_size # get a size of a patch
	pixel_freq_map = {}
	A_new = np.zeros((M,N))
	for patch_id, row in enumerate(X_t):
		i,j = count_patch_coord(patch_id, M, N, patch_size)
		put_patch_back(i, j, row, patch_size, A_new, pixel_freq_map)

	# average all patches for pixels
	for i in range(M):
		for j in range(N):
			try:
				A_new[i][j] /= pixel_freq_map[(i,j)]
			except:
				pass

	return A_new



# patch set method
# make a matrix to count how many pathes are accessing pixel - O(2) lookup time vs 8 if statements (8x complexity)
