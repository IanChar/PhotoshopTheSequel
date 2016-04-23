from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

MASK_VALUE = -1

class ImageHandler(object):

    def __init__(self, path):
        super(ImageHandler, self).__init__()
        self.path = path
        self.img_data = misc.imread(path)
        self.img_shape = (self.img_data.shape[0], self.img_data.shape[1])

    def get_img_data(self):
        return self.img_data

    def render_img_mat(self):
        plt.imshow(self.img_data)
        plt.show()

    def compose_color_mat(self, color):
        color_index = self.find_color_index(color)
        return np.asmatrix(self.img_data[:,:,color_index])

    def compose_grayscale_mat(self):
        to_return = np.asmatrix(np.empty(self.img_shape))
        rows, cols = self.img_shape
        for r in xrange(rows):
            for c in xrange(cols):
                to_return[r, c] = float(sum(list(self.img_data[r, c, :])))/3
        return to_return


    def set_color_matrix(self, color, color_mat):
        color_index = self.find_color_index(color)
        self.img_data[:, :, color_index] = color_mat

    # Note that depending on mask after matrix is masked, picture may not
    # be able to be formed.
    def mask_rgb(self, rgb, tolerance = 150):
        if type(rgb) != tuple or len(rgb) != 3:
            raise ValueError("Not valid rgb")

        rows, cols = self.img_shape
        for r in xrange(rows):
            for c in xrange(cols):
                pixel = tuple(self.img_data[r, c, :])
                diff = abs(rgb[0] - pixel[0]) \
                        + abs(rgb[1] - pixel[1]) \
                        + abs(rgb[2] - pixel[2])
                if diff < tolerance:
                    for i in range(3):
                        self.img_data[r, c, i] = MASK_VALUE


    def find_color_index(self, color):
        colors = ['r', 'g', 'b']
        if color not in colors:
            raise ValueError("Color not recognized.")
        return colors.index(color)


def main():
    ih = ImageHandler("WaterfallWithText.png")
    ih.render_img_mat()

if __name__ == '__main__':
    main()
