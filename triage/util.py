"""
Utility functions.

"""

import numpy as np
from numpy.typing import NDArray


def assert_valid_covariance(matrix: NDArray, tol: float = 1e-8) -> None:
    """
    Raises an error if a matrix is not a valid covariance matrix.

    Parameters:
        matrix array: The matrix to check.
        tol (float): Tolerance for checking symmetry and non-negative
            eigenvalues.
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
