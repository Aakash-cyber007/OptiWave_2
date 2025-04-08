import math

def dot(u, v):
    """Dot product of two vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))

def norm(v):
    """Euclidean norm of a vector."""
    return math.sqrt(dot(v, v))

def frobenius_norm(matrix):
    """Norm of a matrix."""
    total = 0
    for row in matrix:
        for val in row:
            total += val ** 2
    return math.sqrt(total)


def scalar_mult(scalar, v):
    """Multiply vector by scalar."""
    return [scalar * vi for vi in v]

def add_vectors(u, v):
    """add vectors u and v."""
    return [ui - vi for ui, vi in zip(u, v)]

def vector_sub(u, v):
    """Subtract vector v from u."""
    return [ui - vi for ui, vi in zip(u, v)]

def add_matrices(A, B):
    """Adds two matrices element-wise."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions.")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        result.append(row)
    
    return result

def sub_matrices(A, B):
    """Subtracts two matrices element-wise."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions.")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] - B[i][j])
        result.append(row)
    
    return result


def transpose(matrix):
    """Transpose of a matrix."""
    return list(map(list, zip(*matrix)))

def matrix_multiply(A, B):
    # Check if multiplication is possible
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must be equal to number of rows in B.")
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    # Perform multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def matrix_vector_multiply(A, V):
    # Check if multiplication is possible
    if len(A[0]) != len(V):
        raise ValueError("Number of columns in matrix must match size of vector.")
    
    # Initialize result vector
    result = [0 for _ in range(len(A))]

    # Perform multiplication
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j] * V[j]
    
    return result
    
    
def QR(A):
    """Input: Row-major A. Output: Row-major Q, upper triangular R."""
    m = len(A)       # Rows
    n = len(A[0])    # Columns
    Q = [row.copy() for row in A]  # Copy of A (row-major)
    R = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):  # For each column
        # Compute R_ij = q_i • Q_j (q_i = i-th basis vector)
        for i in range(j):
            R[i][j] = sum(Q[k][i] * Q[k][j] for k in range(m))
            for k in range(m):
                Q[k][j] -= R[i][j] * Q[k][i]
        
        # Normalize j-th column of Q
        R[j][j] = math.sqrt(sum(Q[k][j]**2 for k in range(m)))
        if R[j][j] == 0:
            raise ValueError("Matrix is rank-deficient.")
        for k in range(m):
            Q[k][j] /= R[j][j]

    return Q, R
