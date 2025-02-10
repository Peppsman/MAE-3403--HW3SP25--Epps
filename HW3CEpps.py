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

import numpy as np  # Import the numpy library for numerical operations

def is_symmetric(matrix):
    # Check if a matrix is symmetric
    return np.allclose(matrix, matrix.T)

def is_positive_definite(matrix):
    # Check if a matrix is positive definite
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky_solve(A, b):
    # Solve the matrix equation Ax=b using the Cholesky method
    L = np.linalg.cholesky(A)  # Perform Cholesky decomposition
    y = np.linalg.solve(L, b)  # Solve Ly=b
    x = np.linalg.solve(L.T, y)  # Solve L^T x=y
    return x  # Return the solution vector x

def doolittle_solve(A, b):
    # Solve the matrix equation Ax=b using the Doolittle method
    n = len(A)
    L = np.zeros((n, n))  # Initialize L matrix
    U = np.zeros((n, n))  # Initialize U matrix

    # Perform LU Decomposition
    for i in range(n):
        for j in range(i, n):
            L[j][i] = A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))
        for j in range(i, n):
            if i == j:
                U[i][i] = 1
            else:
                U[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))) / L[i][i]

    # Solve Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))


    # Solve Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x  # Return the solution vector x

def gauss_seidel_solve(A, b, tolerance=1e-10, max_iterations=1000):
    # Solve the matrix equation Ax=b using the Gauss-Seidel method
    n = len(A)
    x = np.zeros(n)  # Initialize the solution vector x
    for _ in range(max_iterations):
        x_new = np.copy(x)  # Create a copy of x for updating
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]  # Update the ith element of x
        if np.linalg.norm(x_new - x) < tolerance:  # Check for convergence
            break
        x = x_new  # Update x for the next iteration
    return x  # Return the solution vector x

def main():
    # Main function to check matrix properties and solve Ax=b
    A = np.zeros((4, 4))  # Initialize a 4x4 matrix with zeros
    print("Enter the elements of a 4x4 matrix row by row (or type '1' for preset values):")
    for i in range(4):
        row = input(f"Row {i+1}: ")
        if row.strip() == '1':
            A[i] = np.ones(4)  # Use preset values of 1 for the row
        else:
            A[i] = [float(x) for x in row.split()]  # Convert input to float and assign to the row

    print("Enter the elements of the vector b (or type '1' for preset values):")
    b_input = input()
    if b_input.strip() == '1':
        b = np.ones(4)  # Use preset values of 1 for the vector b
    else:
        b = [float(x) for x in b_input.split()]  # Convert input to float and assign to vector b

    print("Matrix A:")
    print(A)  # Print the matrix A
    print("Vector b:")
    print(b)  # Print the vector b

    if is_symmetric(A) and is_positive_definite(A):
        print("Using Cholesky method:")
        x = cholesky_solve(A, b)  # Solve using the Cholesky method
    else:
        print("Using Doolittle method:")
        x = doolittle_solve(A, b)  # Solve using the Doolittle method

    print("Solution vector x:")
    print(x)  # Print the solution vector x

if __name__ == "__main__":
    main()  # Run the main function when the script is executed
