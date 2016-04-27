import numpy as np
from KSVDSolver import KSVDSolver
from matrix_image import ImageHandler
from matrix_test import encode_image, decode_image, disp_greyscale, add_noise
from sklearn.decomposition import DictionaryLearning


def tryInpainting():
    print "Loading/masking image..."
    ih = ImageHandler("WaterfallWithText.png")
    ih.img_data = ih.img_data[400:500, 50:150, :]
    ih.img_shape = (ih.img_data.shape[0], ih.img_data.shape[1])
    ih.mask_rgb((255, 0, 0))
    mat = ih.compose_grayscale_mat()
    ih.set_color_matrix('r', mat)
    ih.set_color_matrix('g', mat)
    ih.set_color_matrix('b', mat)
    ih.render_img_mat()


    print "Encoding X..."
    mat, m, n = encode_image(mat, patch_size)
    print type(mat)


    print "Learning representation..."

    ksvd = KSVDSolver(mat, masking = True)
    ksvd.learn_dictionary(10)
    dic, code = ksvd.get_representation()

    print "Decoding matrix..."
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


def tryDenoising():
    print "Loading/masking image..."
    ih = ImageHandler("WaterfallWithText.png")
    ih.img_data = ih.img_data[:50, :50, :]
    ih.img_shape = (ih.img_data.shape[0], ih.img_data.shape[1])
    mat = ih.compose_grayscale_mat()
    ih.set_color_matrix('r', mat)
    ih.set_color_matrix('g', mat)
    ih.set_color_matrix('b', mat)
    ih.render_img_mat()


    print "Encoding X..."
    mat, m, n = encode_image(mat)

    print "Learning representation..."
    mat = np.asmatrix(mat).T
    ksvd = KSVDSolver(mat, masking = False)
    ksvd.learn_dictionary(12)
    dic, code = ksvd.get_representation()
    
    print "Decoding matrix..."
    guess = dic * code
    
    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         print mat[i, j], guess[i, j]

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
