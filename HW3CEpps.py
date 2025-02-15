#MAE 3403 HW3CSP25 Epps, Patrick (PBE)


"""
hw3c.py

This program checks if a matrix is symmetric and positive definite.
If the matrix is symmetric and positive definite, the Cholesky method
is used to solve the matrix equation Ax=b. If not, the Doolittle method
is used. The program requests an input for a 4x4 matrix and a vector b,
with preset values of 1 available. The solution vectors are printed
nicely along with an indication of which numerical method was used.
"""

#I used Copilet to create most of this.
#import
#region 

import math
#endregion
#symmetric
#region 

def is_symmetric(matrix):
    """
    Check if a matrix is symmetric.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True
#endregion

#Positive
#region
def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.
    """
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] <= 0:
            return False
        for j in range(i):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True
#endregion

#Cholesky
#region

def cholesky_decomposition(matrix):
    """
    Perform the Cholesky decomposition of a matrix.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                L[i][j] = math.sqrt(matrix[i][i] - sum_k)
            else:
                L[i][j] = (matrix[i][j] - sum_k) / L[j][j]

    return L
#endregion
#Doolittle
#region

def doolittle_decomposition(matrix):
    """
    Perform the Doolittle decomposition of a matrix (LU factorization).
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U
#endregion

#Gauss Seidel
#region

def gauss_seidel(matrix, b, tol=1e-10, max_iter=1000):
    """
    Solve the matrix equation Ax = b using the Gauss-Seidel method.
    """
    n = len(matrix)
    x = [0.0] * n

    for _ in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum_j = sum(matrix[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_j) / matrix[i][i]

        if math.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) < tol:
            return x_new

        x = x_new

    return x
#endregion

#solve all this crazyness
#region
def solve_cholesky(matrix, b):
    """
    Solve the matrix equation Ax = b using the Cholesky decomposition.
    """
    L = cholesky_decomposition(matrix)
    n = len(L)

    # Forward substitution to solve Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]

    # Backward substitution to solve L^T x = y
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]

    return x


def solve_doolittle(matrix, b):
    """
    Solve the matrix equation Ax = b using the Doolittle decomposition.
    """
    L, U = doolittle_decomposition(matrix)
    n = len(L)

    # Forward substitution to solve Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]

    # Backward substitution to solve Ux = y
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]

    return x
#endregion

#Main function 
#region
def main():
    """
    Main function to interact with the user, obtain inputs, and solve matrix equations.
    """
    n = int(input("Enter the dimension of the matrix:  "))
    matrix = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    print("Enter the elements of the matrix row-wise:")
    for i in range(n):
        row = input().split()
        if len(row) != n:
            print(f"Error: Expected {n} elements for row {i + 1}, but got {len(row)} elements.")
            return
        matrix[i] = list(map(float, row))

    print("Enter the elements of the vector b:")
    b = list(map(float, input().split()))
    if len(b) != n:
        print(f"Error: Expected {n} elements for vector b, but got {len(b)} elements.")
        return

    if is_symmetric(matrix) and is_positive_definite(matrix):
        print("Using Cholesky method...")
        x = solve_cholesky(matrix, b)
    else:
        print("Using Doolittle method...")
        x = solve_doolittle(matrix, b)

    print("Solution vector x:")
    for val in x:
        print(val)


if __name__ == "__main__":
    main()
#endregion
