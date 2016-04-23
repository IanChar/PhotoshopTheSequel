def count_patch_coord(patch_id, M, N, patch_size): 
	'''	count the coordinate (i,j) that the first element of 
	patch is pointing to in the orignal image

	Input:
		patch_id - number of patch in X'
		M,N - dimension of our image MxN
	'''
	i = int(patch_id/(M - (patch_size - 1))) # get row number, ex: 0'th row is any patch < M - (n-1)
	j = patch_id % (N - (patch_size - 1)) # get column number
	return i,j

def put_patch_back(base_i, base_j, row, patch_size):
	for z in range(len(row)):
		i = base_i + int(z/patch_size)
		j = base_j + (z % patch_size)
		image[i][j] += row[z]

def assemble_image(X_t):
	#patch_size = self.patch_size # get a size of a patch
	for patch_id, row in enumerate(X_t):
		i,j = count_patch_coord(patch_id, M, N)
		put_patch_back(i, j, row, patch_size)

	# average all patches for pixels
	for i in range(M):
		for j in range(N):
			image[i][j] /= (patch_size*patch_size)

# patch set method
# make a matrix to count how many pathes are accessing pixel - O(2) lookup time vs 8 if statements (8x complexity)
