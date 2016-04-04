import numpy as np
import sklearn as skl
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP

MAX_ITERATIONS = 12

def k_svd(Y, D):
    global MAX_ITERATIONS
    
    J = 1
    X = None
    while J < MAX_ITERATIONS:
        X = sparse_code(Y, D, X)
        update_dictionary(Y, D, X)
        J += 1
    
     

"""
Does sparse coding for a given vector in X, Y, and a dictionary with the 
Orthogonal Matching Pursuit algorithm. Updates each column vector of X to better 
fit Y. Returns adjusted X(Phase 1)
"""

def sparse_code(Y, D, X = None):
    if X is None:
        y_cols, d_cols = Y.shape[1], D.shape[1]
        X = np.asmatrix(np.empty((d_cols, y_cols))
    
    x_rows, x_cols = X.shape
    
    for k in range(x_cols):
        omp = OMP()
        omp.fit(D, y[:, k])
        X[:,k] = np.asmatrix(omp.coef_).T
        
    return X


"""
Forms a matrix for a given vector x to enforce that the new update x will be
sparse. Here N is the columns of Y. Returns the matrix omega.
"""
def form_omega(x, N):
    w = []
    for i, x_i in enumerate(np.nditer(x)):
        if abs(x_i) > 0:
            w.append((i, x_i))
    
    W = np.asmatrix(np.zeroes((N, len(w))
    for w_i, i in w:
        W[w_i, i] = 1
        
    return W
    

"""
Update the dictionary D and the matrix X (phase 2)
"""
def update_dictionary(Y, D, X):
    n, K = D.shape
    # Dhat = np.asmatrix(np.zeroes((n, K)))
    
    # Form E_k
    for k in range(K):
        j = 0
        
        while j < K:
            if j != k:
                E_k = Y - D[:,j]*X[j,:]
                j += 1
            else:
                j += 1
        
        # Form E_kr to ensure that the update will be sparse. Call form_omega
        omega_k = form_omega(X[k,:])
        E_kr = E_k * omega_k
        
        # Form SVD of E_kr and update matrices
        U, sig, V = np.linalg.svd(E_kr, full_matrices = True)
        
        x_kr = sig[0, 0]*V[0,:]
        # Dhat[k,:] = U[0,:]
        D[k,:] = U[0,:]
    
    # Dhat = D
    
def main():
	pass

if __name__ == '__main__':
	main()

  