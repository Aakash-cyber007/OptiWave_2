import random
from .tools import frobenius_norm, sub_matrices, transpose, matrix_multiply, QR

def Block_SVD(A, s = None):
    n, m = len(A), len(A[0])
    # n is no of rows in matrix A.
    if s is None:
        s = n 
    # V is a random block of s vectors each of size n.
    
    V = [[random.randint(0, 99) for _ in range(s)] for _ in range(n)]
    V = list(V)
    def one_iteration(A, V):

        B = matrix_multiply(A, V)
        Q, R = QR(B)
        U = [row[:s] for row in Q] 

        C = matrix_multiply(transpose(A), U)
        Q, R = QR(C)
        V = [row[:s] for row in Q]
        sig = [row[:s] for row in R[:s]]
        return V, U, sig
    
    V, U, sig = one_iteration(A, V)
    error = frobenius_norm(sub_matrices(matrix_multiply(A, V), matrix_multiply(U, sig)))

    # iterating upto convergence
    while(error >= 1e-6):
        V, U, sig = one_iteration(A, V)
        error = frobenius_norm(sub_matrices(matrix_multiply(A, V), matrix_multiply(U, sig)))
    # at convergence, we get M as a diagonal matrix.
    # diagonal elements of M are singular values are first s singular values of A.
    S = []
    for i in range(s):
        S.append(sig[i][i])
    return U, S, transpose(V)
