import numpy as np
from KSVDSolver import KSVDSolver
from matrix_image import ImageHandler
from matrix_test import encode_image, decode_image, disp_greyscale

def main():
    patch_size = 19
    ih = ImageHandler("WaterfallWithText.png")
    ih.img_data = ih.img_data[50:70, 50:70, :]
    ih.img_shape = (ih.img_data.shape[0], ih.img_data.shape[1])
    mat = ih.compose_grayscale_mat()

    print "Encoding X..."
    mat, m, n = encode_image(mat, patch_size)
    print type(mat)

    print "Learning representation..."
    mat = np.asmatrix(mat).T
    ksvd = KSVDSolver(mat)
    ksvd.learn_dictionary(3)
    dic, code = ksvd.get_representation()

    print mat.shape, dic.shape, code.shape
    guess = dic * code
    ''''
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print mat[i, j], guess[i, j]
    '''
    guess = np.asarray(guess.T)

    result = decode_image(guess, m, n, patch_size)

    ih.set_color_matrix('r', result)
    ih.set_color_matrix('g', result)
    ih.set_color_matrix('b', result)
    ih.render_img_mat()
    ih.render_img_mat_compare()

if __name__ == '__main__':
    main()
