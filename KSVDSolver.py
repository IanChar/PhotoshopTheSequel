from __future__ import division
import numpy as np
import numpy.linalg as linalg
import sklearn.decomposition as Decomp
from sklearn.linear_model import OrthogonalMatchingPursuit as omp
from matrix_image import MASK_VALUE

class KSVDSolver(object):
    """ KSVDSolver
    - NOTE: All matrices used should be numpy.matrix

    Attributes
    ----------
    self.signals:  is the matrix Y that holds the signals of the data where
        each column is a particular signal. For images, each of the columns
        represents some patch of the picture.

    self.encoding: is the matrix X that holds the current sparse encodings
        of the signals in Y using the current dictionary.

    self.dictionary: is the current overcomplete dictionary that we are using
        to represent the signals

    self.total_iterations: represents the total amount of iterations we
        have done of KSVD to get the current dictionary
    """

    def __init__(self, signals, masking = False, dictionary = None, encoding = None):
        super(KSVDSolver, self).__init__()
        self.signals = signals
        self.masking = masking

        # Set to -1 now so that initializers will know
        self.total_iterations = -1

        if dictionary is None:
            self.find_init_dict()
        else:
            self.dictionary = dictionary

        self.encoding = encoding
        if encoding is None:
            self.sparse_encode(masking)

        # Now that everything has been initialized we set iterations to 0
        self.total_iterations = 0

    def learn_dictionary(self, iterations = 10):
        for _ in range(iterations):
            self.sparse_encode(self.masking)
            # print self.get_error()
            self.update_dictionary()
        self.sparse_encode(self.masking)

    def get_representation(self):
        return self.dictionary, self.encoding

    def get_error(self, norm = 'fro'):
        return linalg.norm(self.signals - self.dictionary * self.encoding, norm)

    """ HELPER FUNCTIONS """

    def sparse_encode(self, masking, nonzero_coefs = None, pursuit = None):
        if pursuit is None:
            pursuit = omp(n_nonzero_coefs = nonzero_coefs)

        if self.encoding is None:
            self.encoding = np.asmatrix(np.empty((self.dictionary.shape[1], \
                    self.signals.shape[1])))

        for c in xrange(self.signals.shape[1]):
            if masking:
                maskedVect = np.copy(self.signals[:,c])
                maskedDic = np.copy(self.dictionary)
                for r in xrange(self.signals.shape[0]):
                    if maskedVect[r, 0] == MASK_VALUE:
                        maskedVect[r, 0] = 0
                        maskedDic[r, :] = np.asmatrix(np.zeroes(1, maskedDic.shape[1]))
                pursuit.fit(maskedDic, maskedVect)
            else:
                pursuit.fit(self.dictionary, self.signals[:, c])
            self.encoding[:, c] = np.asmatrix(pursuit.coef_).T

        return self.encoding

    def update_dictionary(self):
        errorMat = self.signals - self.dictionary * self.encoding
        for k in range(self.dictionary.shape[1]):
            # Compute SVD for the sparse elements
            updatedError = errorMat \
                    + self.dictionary[:, k] * self.encoding[k, :]
            updatedError = self.throw_out_sparseness(updatedError, \
                    self.encoding[k, :])
            try:
                U, S, V = linalg.svd(updatedError)
            except linalg.LinAlgError:
                continue

            # Update dictionary and representation
            self.dictionary[:, k] = U[:, 0]



    # The initial dictionary we start with if no alternative is provided
    # is the U of the SVD.
    def find_init_dict(self):
        if self.total_iterations != -1:
            print "The dictionary has already been inqitialized!"
            return
        U, S, V = linalg.svd(self.signals, full_matrices=False)
        self.dictionary = self.normalize_columns(np.asmatrix(U))
        return self.dictionary


    def normalize_columns(self, mat, norm = 2):
        for c in xrange(mat.shape[1]):
            mat[:, c] /= linalg.norm(mat[:, c], norm)
        return mat

    # Assuming the sparse_vector is a row vector
    def throw_out_sparseness(self, error_mat, sparse_vector):
        vect_rows = sparse_vector.shape[1]
        # Form Omega from the paper
        nonsparse_indices = []
        for i in range(vect_rows):
            if sparse_vector[0, i] != 0:
                nonsparse_indices.append(i)

        omega = np.asmatrix(np.zeros((vect_rows, len(nonsparse_indices))))
        for i, w_i in enumerate(nonsparse_indices):
            omega[w_i, i] = 1

        return error_mat * omega

def compareToScikit(iterations):
    original = np.asmatrix(np.random.rand(10,53))
    print linalg.norm(original, 'fro')

    # Our ksvd
    ksvd = KSVDSolver(original)
    ksvd.learn_dictionary(iterations)
    print ksvd.get_error()

    # Scikit learn's dictionary learning
    dl = Decomp.DictionaryLearning()
    ret = dl.fit_transform(original)
    ans = np.asmatrix(ret) * np.asmatrix(dl.components_).T
    print linalg.norm(original - ans, 'fro')


def test():
    original = np.asmatrix(np.random.rand(10,53))
    ksvd = KSVDSolver(original)
    print linalg.norm(original, 'fro')
    ksvd.learn_dictionary(100)


if __name__ == '__main__':
    compareToScikit(10)
