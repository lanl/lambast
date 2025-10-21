"""
Generate complex-valued NQR signals using Voigt model.

Authors: Natalie Klein, Amber Day
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import triage.util as util
from numpy.typing import NDArray
from scipy.special import wofz


@dataclass
class Voigt(object):
    """
    Class to manage VoigtSignal and associated functions

    Inputs
        sample_n : number of examples
        nt : number of time points
        fs : sampling rate
        freq_range : upper and lower bounds for the frequency, freq
        phi_range : upper and lower bounds for the phase offset, phi
        decay_rate_rage: upper and lower bounds for the decay_rate
        sigma_range : upper and lower bounds for the sigma
        amp_range : upper and lower bounds for the initial
            amplitude, amp
        noise_var : variance for white noise generation
        seed: random seed
        f_vec: frequency vector for freqs
        t : corresponding time vector for sigs
    """

    sample_n: int = 10000
    nt: int = 1024
    fs: float = 1.0 / 1.8e-05
    freq_range: tuple[float, float] = (-50, 50)
    phi_range: tuple[float, float] = (-np.pi, np.pi)
    decay_rate_range: tuple[float, float] = (1e-3, 1e-2)
    sigma_range: tuple[float, float] = (1e-3, 1e-2)
    amp_range: tuple[float, float] = (0.1, 1)
    noise_var: float = 1
    rng: np.random.Generator = np.random.default_rng()

    def synthetic_data_gen(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Generating synthetic data with white noise

        Outputs
            df_in: df containing N, nt, fs, w, phi, T2, A, sigma,
                s, C, t, f_vec
            df: df containing sigs, noise, noisy_sigs, freqs
            sigs: time series voigt signals
            noise: time series white noise
            freqs: frequency series voigts signals
        """

        # Initializing Parameters
        freq = self.rng.uniform(
            self.freq_range[0], self.freq_range[1], self.sample_n)
        phi = self.rng.uniform(
            self.phi_range[0], self.phi_range[1], self.sample_n)
        decay_rate = self.rng.uniform(
            self.decay_rate_range[0], self.decay_rate_range[1], self.sample_n)
        sigma = self.rng.uniform(
            self.sigma_range[0], self.sigma_range[1], self.sample_n)
        amp = self.rng.uniform(
            self.amp_range[0], self.amp_range[1], self.sample_n)
        const = np.zeros(self.sample_n)
        df = pd.DataFrame({"freq": freq, "phi": phi, "decay_rate": decay_rate,
                          "amp": amp, "sigma": sigma, "const": const})

        # Signal Generation
        t = np.linspace(0, self.nt / self.fs, self.nt)
        f_vec = np.linspace(-500, 500, self.nt)
        sigs = self.sig_gen(t, df)
        freqs = self.freq_gen(f_vec, df)

        # Generate White Noise
        noise = util.white_noise_gen(
            t, self.sample_n, sigma=self.noise_var, rng=self.rng)
        snrs = util.compute_snr(sigs, noise)
        noisy_sig = sigs + noise
        df_in = {
            "N": self.sample_n,
            "nt": self.nt,
            "fs": self.fs,
            "freq": freq,
            "phi": phi,
            "decay_rate": decay_rate,
            "amp": amp,
            "sigma": sigma,
            "noise_var": self.noise_var,
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

    def __freq_or_sig_gen(self, vector: NDArray[np.float64],
                          df: pd.DataFrame, mode: str
                          ) -> NDArray[np.float64] | NDArray[np.complex64]:
        """
        Function factoring out the common parts of sig_gen and freq_gen
        Args:
            t: vector of time or frequency points
            df: Pandas data frame containing signal parameter columns amp,
                freq, decay_rate, phi, sigma, const, with N rows
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
                                 df.amp.iloc[i], df.phi.iloc[i],
                                 df.sigma.iloc[i], df.const.iloc[i])
            if mode == "time":
                result[i, :] = signal.time_signal(vector)
            elif mode == "freq":
                result[i, :] = signal.freq_signal(vector)

        return result

    def sig_gen(self, t: NDArray, df: pd.DataFrame) -> NDArray:
        """
        Generate collection of Voigt time series signals based on Pandas data
        frame of parameter values.
        Args:
            t: vector of time points
            df: Pandas data frame containing signal parameter columns amp,
                freq, decay_rate, phi, sigma, const, with N rows
        Returns:
            np.ndarray of signals, shape (N, len(t))
        """

        return self.__freq_or_sig_gen(t, df, mode="time")

    def freq_gen(self, f_vec: NDArray, df: pd.DataFrame) -> NDArray:
        """
        Generate collection of Voigt frequency signals based on Pandas
        data frame of parameter values.
        Args:
            f_vec: vector of frequency points
            df: Pandas data frame containing signal parameter columns amp,
                freq, decay_rate, phi, sigma, const, with N rows
        Returns:
            np.ndarray of signals, shape (N, len(f_vec))
        """

        return self.__freq_or_sig_gen(f_vec, df, mode="freq")


@dataclass
class VoigtSignal(object):
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
        fit = np.real(wofz(exp_arg)) / self.sigma

        return self.amp * fit / np.max(fit)
