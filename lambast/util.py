"""
Utility functions.

"""

import matplotlib.pyplot as plt
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


def compute_snr(sig: NDArray[np.complex64],
                noise: NDArray[np.complex64]) -> NDArray[np.complex64]:
    """
    Compute SNR empirically given signal and noise data.
    """

    noise_var = np.var(noise - np.mean(noise, 1, keepdims=True), 1)
    sig_var = np.var(sig - np.mean(sig, 1, keepdims=True), 1)

    return 10.0 * np.log10(sig_var / noise_var)


def white_noise_gen(t: NDArray, N: int, sigma: float = 1.0,
                    rng: np.random.Generator = np.random.default_rng()
                    ) -> NDArray:
    """
    Generate complex Gaussian white noise time series of shape (N, t) with
    total variance sigma.
    """

    r_dist1 = rng.normal(0, scale=sigma / np.sqrt(2), size=(N, len(t)))
    i_dist2 = rng.normal(0, scale=sigma / np.sqrt(2), size=(N, len(t)))

    return r_dist1 + 1j * i_dist2


def split_noise(noise, n_timesteps):
    """
    Split noise data array along time axis to generate a higher number of
    shorter examples
    """

    # TEST: does this work for an edge case
    # where (noise.shape[1] % n_timesteps) == 0?
    nseg = noise.shape[1] // n_timesteps
    noise = noise[:, : (nseg * n_timesteps)]
    snoise = np.stack(np.hsplit(noise, nseg), axis=1)
    rsnoise = np.reshape(snoise, (noise.shape[0] * nseg, -1))

    return rsnoise


def plot_complex_ts(t, y, ax=None, **kwargs):
    """
    Plot complex-valued time series
    """

    if ax is None:
        ax = plt.gca()
    for f, c in zip([np.real, np.imag], ["k", "r"]):
        ax.plot(t, f(y), c, **kwargs, label=f.__name__)
