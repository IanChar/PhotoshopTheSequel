import numpy as np
from KSVDSolver import KSVDSolver
from matrix_image import ImageHandler
from matrix_test import encode_image, decode_image, disp_greyscale, add_noise
from sklearn.decomposition import DictionaryLearning

PATCH_SIZE = 8

def tryDenoising():
    print "___________LOADING IN IMAGE______________"
    ih = ImageHandler("downsized_falls.png")
    ih.add_gaussian_noise(variance= 50)
    mat = ih.compose_grayscale_mat()
    ih.set_color_matrix('r', mat)
    ih.set_color_matrix('g', mat)
    ih.set_color_matrix('b', mat)
    ih.render_img_mat()


    print "________________UNROLLING X______________"
    mat, m, n = encode_image(mat, PATCH_SIZE)

    print "___________________KSVD____________________"
    mat = np.asmatrix(mat).T
    ksvd = KSVDSolver(mat, masking = False)
    ksvd.learn_dictionary(3)
    dic, code = ksvd.get_representation()
    guess = dic * code
    guess = np.asarray(guess.T)

    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         print mat[i, j], guess[i, j]


    print "_____________ROLLING UP X_________________"

    result = decode_image(guess, m, n, PATCH_SIZE)
    ih.set_color_matrix('r', result)
    ih.set_color_matrix('g', result)
    ih.set_color_matrix('b', result)
    ih.render_img_mat()

def main():
    tryDenoising()


if __name__ == '__main__':
    main()
