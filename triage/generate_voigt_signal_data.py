"""
Generate complex-valued NQR signals using Voigt model.

Authors: Natalie Klein, Amber Day
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import wofz


@dataclass
class VoigtSignal:
    """
    Voigt signal model. If sigma=np.inf, recover FID signal model.
    Args:
        freq: frequency (Hz)
        decay_rate: time decay constant
        amp: amplitude constant
        phi: phase shift constant
        sigma: shape constant
        const: constant shift
    """

    freq: float
    decay_rate: float
    amp: float
    phi: float
    sigma: float
    const: float

    def time_signal(self, t: NDArray[np.float64]) -> NDArray[np.complex64]:
        """
        Return complex-valued time-domain signal given time vector t.
        """

        exp1 = np.exp(-0.5 * (t / self.sigma) ** 2)
        exp2 = np.exp(-t / self.decay_rate)
        exp3 = np.exp(1j * (2 * np.pi * self.freq * t + self.phi))

        return self.amp * exp1 * exp2 * exp3 + self.const

    def freq_signal(self, f_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Return real-valued frequency-domain representation given
        frequency vector f_vec.
        """

        exp_arg = (f_vec - self.freq + 1j * self.decay_rate)
        exp_arg /= (self.sigma * np.sqrt(2))
        fit = self.amp * np.real(wofz(exp_arg)) / self.sigma

        return fit / np.max(fit)


def compute_snr(sig: NDArray[np.complex64],
                noise: NDArray[np.complex64]) -> NDArray[np.complex64]:
    """
    Compute SNR empirically given signal and noise data.
    """

    noise_var = np.var(noise - np.mean(noise, 1, keepdims=True), 1)
    sig_var = np.var(sig - np.mean(sig, 1, keepdims=True), 1)

    return 10.0 * np.log10(sig_var / noise_var)


def freq_or_sig_gen(vector: NDArray[np.float64], df: pd.DataFrame,
                    mode: str) -> NDArray[np.float64] | NDArray[np.complex64]:
    """
    Function factoring out the common parts of sig_gen and freq_gen
    Args:
        t: vector of time or frequency points
        df: Pandas data frame containing signal parameter columns amp, freq,
            decay_rate, phi, sigma, const, with N rows
        mode: "time" or "freq", for the type of return
    Returns:
        np.ndarray of signals, shape (N, len(vector))
    """

    assert mode in ["time", "freq"]
    dtype: type[complex] | type[float]

    if mode == "time":
        dtype = complex
    elif mode == "freq":
        dtype = float

    result = np.zeros((df.shape[0], len(vector)), dtype=dtype)
    for i in range(df.shape[0]):
        signal = VoigtSignal(df.freq.iloc[i], df.decay_rate.iloc[i],
                             df.amp.iloc[i], df.phi.iloc[i], df.sigma.iloc[i],
                             df.const.iloc[i])
    if mode == "time":
        result[i, :] = signal.time_signal(vector)
    elif mode == "freq":
        result[i, :] = signal.freq_signal(vector)

    return result


def sig_gen(t: NDArray, df: pd.DataFrame) -> NDArray:
    """
    Generate collection of Voigt time series signals based on Pandas data
    frame of parameter values.
    Args:
        t: vector of time points
        df: Pandas data frame containing signal parameter columns amp, freq,
            decay_rate, phi, sigma, const, with N rows
    Returns:
        np.ndarray of signals, shape (N, len(t))
    """

    return freq_or_sig_gen(t, df, mode="time")


def freq_gen(f_vec: NDArray, df: pd.DataFrame) -> NDArray:
    """
    Generate collection of Voigt frequency signals based on Pandas data frame
    of parameter values.
    Args:
        f_vec: vector of frequency points
        df: Pandas data frame containing signal parameter columns amp, freq,
            decay_rate, phi, sigma, const, with N rows
    Returns:
        np.ndarray of signals, shape (N, len(f_vec))
    """

    return freq_or_sig_gen(f_vec, df, mode="freq")


def white_noise_gen(t: NDArray, N: int, sigma: float = 1.0) -> NDArray:
    """
    Generate complex Gaussian white noise time series of shape (N, t) with
    total variance sigma.
    """

    r_dist1 = np.random.normal(0, scale=sigma / np.sqrt(2), size=(N, len(t)))
    i_dist2 = np.random.normal(0, scale=sigma / np.sqrt(2), size=(N, len(t)))

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


def synthetic_data_gen(N: int = 10000, nt: int = 1024,
                       fs: float = 1.0 / 1.8e-05,
                       freq_range: tuple[float, float] = (-50, 50),
                       phi_range: tuple[float, float] = (-np.pi, np.pi),
                       decay_rate_range: tuple[float, float] = (1e-3, 1e-2),
                       sigma_range: tuple[float, float] = (1e-3, 1e-2),
                       amp_range: tuple[float, float] = (0.1, 1),
                       noise_var: float = 1,
                       seed: int = 42,
                       f_vec: NDArray = np.linspace(-100, 100, 1000)
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Generating synthetic data with white noise

    Inputs
        N : number of examples
        nt : number of time points
        fs : sampling rate
        freq_range : upper and lower bounds for generating the frequency, freq
        phi_range : upper and lower bounds for generating the phase offset, phi
        decay_rate_rage: upper and lower bounds for generating the decay_rate
        sigma_range : upper and lower bounds for generating the sigma
        amp_range : upper and lower bounds for generating the initial
            amplitude, amp
        noise_var : variance for white noise generation
        seed: random seed
        f_vec: frequency vector for freqs
        t : corresponding time vector for sigs
    Outputs
        df_in: df containing N, nt, fs, w, phi, T2, A, sigma, s, C, t, f_vec
        df: df containing sigs, noise, noisy_sigs, freqs
        sigs: time series voigt signals
        noise: time series white noise
        freqs: frequency series voigts signals
    """

    # Initializing Parameters
    np.random.seed(seed)
    freq = np.random.uniform(freq_range[0], freq_range[1], N)
    phi = np.random.uniform(phi_range[0], phi_range[1], N)
    decay_rate = np.random.uniform(decay_rate_range[0], decay_rate_range[1], N)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], N)
    amp = np.random.uniform(amp_range[0], amp_range[1], N)
    const = np.zeros(N)
    df = pd.DataFrame({"freq": freq, "phi": phi, "decay_rate": decay_rate,
                      "amp": amp, "sigma": sigma, "const": const})

    # Signal Generation
    t = np.linspace(0, nt / fs, nt)
    f_vec = np.linspace(-500, 500, nt)
    sigs = sig_gen(t, df)
    freqs = freq_gen(f_vec, df)

    # Generate White Noise
    noise = white_noise_gen(t, N, sigma=noise_var)
    snrs = compute_snr(sigs, noise)
    noisy_sig = sigs + noise
    df_in = {
        "N": N,
        "nt": nt,
        "fs": fs,
        "freq": freq,
        "phi": phi,
        "decay_rate": decay_rate,
        "amp": amp,
        "sigma": sigma,
        "noise_var": noise_var,
        "const": const,
        "t": t,
        "f_vec": f_vec,
    }

    df_out = {
        "snr": snrs,
        "sigs": sigs,
        "noise": noise,
        "noisy_sig": noisy_sig,
        "freqs": freqs,
    }

    return df_in, df_out


def plot_complex_ts(t, y, ax=None, **kwargs):
    """
    Plot complex-valued time series
    """

    if ax is None:
        ax = plt.gca()
    for f, c in zip([np.real, np.imag], ["k", "r"]):
        ax.plot(t, f(y), c, **kwargs, label=f.__name__)
