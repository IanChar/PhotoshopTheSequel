import numpy as np
from KSVDSolver import KSVDSolver
from matrix_image import ImageHandler
from matrix_test import encode_image, decode_image, disp_greyscale, add_noise
from sklearn.decomposition import DictionaryLearning

def tryDenoising():
    print "___________LOADING IN IMAGE______________"
    ih = ImageHandler("WaterfallWithText.png")
    ih.add_gaussian_noise(variance= 100)
    ih.img_data = ih.img_data[:100, :100, :]
    ih.img_shape = (ih.img_data.shape[0], ih.img_data.shape[1])
    mat = ih.compose_grayscale_mat()
    ih.set_color_matrix('r', mat)
    ih.set_color_matrix('g', mat)
    ih.set_color_matrix('b', mat)
    ih.render_img_mat()


    print "________________UNROLLING X______________"
    mat, m, n = encode_image(mat)

    print "___________________KSVD____________________"
    mat = np.asmatrix(mat).T
    ksvd = KSVDSolver(mat, masking = False)
    ksvd.learn_dictionary(3)
    dic, code = ksvd.get_representation()

    print "_____________ROLLING UP X_________________"
    guess = dic * code

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print mat[i, j], guess[i, j]

    """ Try Scikit's built in method"""
    # dl = DictionaryLearning()
    # code = dl.fit_transform(mat)
    # guess = code * dl.components_

    guess = np.asarray(guess.T)

    result = decode_image(guess, m, n)
    ih.set_color_matrix('r', result)
    ih.set_color_matrix('g', result)
    ih.set_color_matrix('b', result)
    ih.render_img_mat()

def main():
    tryDenoising()


if __name__ == '__main__':
    main()
