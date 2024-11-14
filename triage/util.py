"""
Utility functions.

"""

import numpy as np


def is_valid_covariance(matrix, tol=1e-8):
    """
    Checks if a matrix is a valid covariance matrix.

    Parameters:
        matrix (np.ndarray): The matrix to check.
        tol (float): Tolerance for checking symmetry and non-negative
            eigenvalues.

    Returns:
        bool: True if the matrix is a valid covariance matrix, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix is not square.")

    # Check symmetry
    if not np.allclose(matrix, matrix.T, atol=tol):
        raise ValueError("Covariance matrix is not symmetric.")

    # Check positive semi-definiteness using eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(eigenvalues < -tol):
        raise ValueError("Covariance matrix is not positive semi-definite.")
