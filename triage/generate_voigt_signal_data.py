"""
Generate complex-valued NQR signals using Voigt model.

Authors: Natalie Klein, Amber Day
"""

import numpy as np
import pandas as pd
from scipy.special import wofz
import matplotlib.pyplot as plt


class VoigtSignal:
    """
    Voigt signal model.
    """

    def __init__(self, w, T2, A, phi, sigma, C):
        """
        Voigt signal model. If sigma=np.inf, recover FID signal model.
        Args: 
            w: frequency (Hz)
            T2: time decay constant
            A: amplitude constant
            phi: phase shift constant
            sigma: shape constant
            C: constant shift
        """
        super(VoigtSignal, self).__init__()
        self.w = w
        self.T2 = T2
        self.A = A
        self.phi = phi
        self.sigma = sigma
        self.C = C

    def time_signal(self, t):
        """
        Return complex-valued time-domain signal given time vector t.
        """
        return (
            self.A
            * np.exp(-0.5 * (t / self.sigma) ** 2)
            * np.exp(-t / self.T2)
            * np.exp(1j * (2 * np.pi * self.w * t + self.phi))
            + self.C
        )

    def freq_signal(self, f_vec):
        """
        Return real-valued frequency-domain representation given frequency vector f_vec.
        """
        fit = (
            np.real(wofz((f_vec - self.w + 1j * self.T2) / self.sigma / np.sqrt(2)))
            / self.sigma
        )
        return self.A * fit / np.max(fit)


def compute_snr(sig, noise):
    """
    Compute SNR empirically given signal and noise data.
    """
    noise_var = np.var(noise - np.mean(noise, 1, keepdims=True), 1)
    sig_var = np.var(sig - np.mean(sig, 1, keepdims=True), 1)
    return 10.0 * np.log10(sig_var / noise_var)


def sig_gen(t, df):
    """
    Generate collection of Voigt time series signals based on Pandas data frame of parameter values.
    Args:
        t: vector of time points
        df: Pandas data frame containing signal parameter columns A, w, T2, phi, sigma, C, with N rows
    Returns:
        np.ndarray of signals, shape (N, len(t))
    """
    sigs = np.zeros((df.shape[0], len(t)), dtype=complex)
    for i in range(df.shape[0]):
        sigs[i, :] = VoigtSignal(
            df.w.iloc[i],
            df.T2.iloc[i],
            df.A.iloc[i],
            df.phi.iloc[i],
            df.sigma.iloc[i],
            df.C.iloc[i],
        ).time_signal(t)
    return sigs


def wn_gen(t, N, sigma=1.0):
    """
    Generate complex Gaussian white noise time series of shape (N, t) with total variance sigma.
    """
    return np.random.normal(
        0, scale=sigma / np.sqrt(2), size=(N, len(t))
    ) + 1j * np.random.normal(0, scale=sigma / np.sqrt(2), size=(N, len(t)))


def split_noise(noise, n_timesteps):
    """
    Split noise data array along time axis to increase number of examples (with shorter duration).
    """
    nt = noise.shape[1]
    nseg = nt // n_timesteps
    noise = noise[:, : (nseg * n_timesteps)]
    snoise = np.stack(np.hsplit(noise, nseg), axis=1)
    rsnoise = np.reshape(snoise, (noise.shape[0] * nseg, -1))
    return rsnoise


def freq_gen(f_vec, df):
    """
    Generate collection of Voigt frequency signals based on Pandas data frame of parameter values.
    Args:
        f_vec: vector of frequency points
        df: Pandas data frame containing signal parameter columns A, w, T2, phi, sigma, C, with N rows
    Returns:
        np.ndarray of signals, shape (N, len(f_vec))
    """
    freqs = np.zeros((df.shape[0], len(f_vec)))
    for i in range(df.shape[0]):
        freqs[i, :] = VoigtSignal(
            df.w.iloc[i],
            df.T2.iloc[i],
            df.A.iloc[i],
            df.phi.iloc[i],
            df.sigma.iloc[i],
            df.C.iloc[i],
        ).freq_signal(f_vec)
    return freqs


def synthetic_data_gen(
    N=10000,
    nt=1024,
    fs=1.0 / 1.8e-05,
    w_range=[-50, 50],
    phi_range=[-np.pi, np.pi],
    T2_range=[1e-3, 1e-2],
    sigma_range=[1e-3, 1e-2],
    A_range=[0.1, 1],
    s=1,
    seed=42,
    f_vec=np.linspace(-100, 100, 1000),
):
    """
    Generating synthetic data with white noise
    
    Inputs
        N : number of examples
        nt : number of time points
        fs : sampling rate
        w_range : upper and lower bounds for generating the frequency, w
        phi_range : upper and lower bounds for generating the phase offset, phi
        T2_range & sigma_range : upper and lower bounds for generating the rate of decay, T2 & sigma
        A_range : upper and lower bounds for generating the initial amplitude, A
        s : variance for white noise generation
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
    w = np.random.uniform(w_range[0], w_range[1], N)
    phi = np.random.uniform(phi_range[0], phi_range[1], N)
    T2 = np.random.uniform(T2_range[0], T2_range[1], N)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], N)
    A = np.random.uniform(A_range[0], A_range[1], N)
    C = np.zeros(N)
    df = pd.DataFrame({"w": w, "phi": phi, "T2": T2, "A": A, "sigma": sigma, "C": C})

    # Signal Generation
    t = np.linspace(0, nt / fs, nt)
    f_vec = np.linspace(-500, 500, nt)
    sigs = sig_gen(t, df)
    freqs = freq_gen(f_vec, df)

    # Generate White Noise
    noise = wn_gen(t, N, sigma=s)
    snrs = compute_snr(sigs, noise)
    noisy_sig = sigs + noise
    df_in = {
        "N": N,
        "nt": nt,
        "fs": fs,
        "w": w,
        "phi": phi,
        "T2": T2,
        "A": A,
        "sigma": sigma,
        "s": s,
        "C": C,
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
    Plot complex-valued time series. 
    """
    if ax is None:
        ax = plt.gca()
    for f, c in zip([np.real, np.imag], ["k", "r"]):
        ax.plot(t, f(y), c, **kwargs, label=f.__name__)
