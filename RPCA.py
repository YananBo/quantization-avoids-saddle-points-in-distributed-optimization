import numpy as np
from scipy.linalg import svd
import os
from PIL import Image
import matplotlib.pyplot as plt
# import cupy as cp

def sparse_operator(matrix, alpha = 5):
    """
    Apply a sparse operator that keeps only the top 1/5-fraction largest elements
    in each row and column of the given matrix.

    Parameters:
    - matrix: Input matrix

    Returns:
    - sparse_matrix: Resulting sparse matrix
    """

    # Get the number of rows and columns in the matrix
    num_rows, num_cols = matrix.shape

    # Calculate the number of elements to keep in each row and column (1/5-fraction)
    num_elements_to_keep_row = int(num_cols / alpha)
    num_elements_to_keep_col = int(num_rows / alpha)

    # Initialize the sparse matrix
    sparse_matrix = np.zeros_like(matrix)

    # Apply sparse operator to each row
    for i in range(num_rows):
        row = matrix[i, :]
        indices_to_keep_row = np.argsort(row)[-num_elements_to_keep_row:]
        sparse_matrix[i, indices_to_keep_row] = row[indices_to_keep_row]

    # Apply sparse operator to each column
    for j in range(num_cols):
        col = matrix[:, j]
        indices_to_keep_col = np.argsort(col)[-num_elements_to_keep_col:]
        sparse_matrix[indices_to_keep_col, j] = col[indices_to_keep_col]

    return sparse_matrix

def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), 0)

def initialize_UV_S(D, r=5):
    S_0 = sparse_operator(D, alpha = 5)
    U_0, Sigma_0, V_0 = svd(D - S_0, full_matrices=False)
    # Use the specified rank 'r'
    U_0 = U_0[:, :r]
    Sigma_0 = np.diag(Sigma_0[:r])
    V_0 = V_0[:r, :]

    U = U_0 @ (np.sqrt(np.diag(Sigma_0[:r])))
    V = (np.sqrt(np.diag(Sigma_0[:r]))) @ V_0
    
    L = U_0 @ Sigma_0 @ V_0 
   
    S = sparse_operator(D - L, alpha = 5)
    
    return U_0, V_0, S

def robust_pca_gradient_descent(D, r=30, mu=100, learning_rate=1e-5, max_iter=1000, tol=1e-7, iter_print=100 ):
    m, n = D.shape
    mu_inv = 1 / mu
    beta = 1 / (2 * np.power(m * n, 1/4))
    beta_init = 4 * beta
    U, V, S = initialize_UV_S(D, r)
    # print(U.shape)
    # print(V.shape)
    # print(S.shape)
    for i in range(max_iter):
        
        gradient_U = -2 * (D - np.dot(U, V) - S).dot(V.T) + 2 * mu_inv * U
        gradient_V = -2 * (D - np.dot(U, V) - S).T.dot(U) + 2 * mu_inv * V.T 
              
        U = U - learning_rate * gradient_U
        V = V - learning_rate * (gradient_V).T
           
        S = sparse_operator (D - np.dot(U, V), alpha = 5)
        err = np.linalg.norm(D - np.dot(U, V) - S, ord='fro')
        if (i % iter_print) == 0 or i == 1 or i > max_iter or err <= tol:
                print('iteration: {0}, error: {1}'.format(i, err))
        if err < tol:
            break

    return U, V.T, S

def load_sequence(path, start, stop):
    
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    
    # files = os.listdir(path)
    files.sort()
    frame = Image.open(os.path.join(path, files[1]))
    frame = frame.convert("L")
    frame = np.array(frame)
    frame_size = frame.shape
    size = frame.shape[0] * frame.shape[1]

    M = np.zeros((size, stop - start), np.float32)
    
    for i in range(start, stop):
        frame = Image.open(os.path.join(path, files[i])).convert("L")
        frame = np.array(frame)
        M[:, i] = frame.reshape(-1)
    
    return M, frame_size


videos = [
            "./videos/Bootstrap",
            "./videos/Camouflage",
            "./videos/ForegroundAperture",
            "./videos/LightSwitch",
            "./videos/MovedObject",
            "./videos/TimeOfDay",
            "./videos/WavingTrees"
         ]

start = 0
stop = 200
M, frame_size = load_sequence(videos[0], start, stop)

U, V.T, S = robust_pca_gradient_descent (M)

