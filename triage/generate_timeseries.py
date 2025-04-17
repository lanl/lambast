"""
Time series generation classes.

"""

import copy
import warnings
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import fsolve
from scipy.stats import truncnorm

from .util import assert_valid_covariance


class TimeSeries:

    def __init__(self) -> None:
        """
        TimeSeries base class.

        """

        return None

    def sample(self, n: int, t: int, *args, **kwargs
               ) -> NDArray | tuple[list[NDArray], list[NDArray]]:
        """
        Sample `n` time series with `t` time points from the TimeSeries model.

        Parameters:
            n (int): number of time series replicates to sample
            t (int): number of time points in each time series
        """
        raise NotImplementedError("Subclasses must implement this method.")


class LinearSSM(TimeSeries):

    def __init__(self, state_matrix: NDArray, state_noise_cov: NDArray,
                 obs_matrix: NDArray, obs_noise_cov: NDArray,
                 rng: np.random.Generator | None = None,
                 scale_matrix: bool = False) -> None:
        """
        Initializes a new instance of a Linear State Space Model.
        State dimension is d, observation dimension is p.
        Assume zero-mean Gaussian noise in state and observation space.

        Parameters:
            state_matrix: shape (d,d) numpy array defining state
                transition matrix
            state_noise_cov: shape (d,d) numpy array defining
                state noise covariance
            obs_matrix: shape (p,d) numpy array defining
                observation matrix
            obs_noise_cov: shape (p,p) numpy array defininig
                observation noise covariance
            rng: numpy rng (else use default); see
                https://numpy.org/doc/2.0/reference/random/index.html#random-quick-start
            scale_matrix (bool): Whether to rescale the matrix for stability,
                default, false.
        """
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.d = state_matrix.shape[0]
        self.p = obs_matrix.shape[0]

        self.state_matrix = state_matrix
        self.state_noise_cov = state_noise_cov
        self.obs_matrix = obs_matrix
        self.obs_noise_cov = obs_noise_cov

        # Check if state matrix is stable and warn user if not, but only if
        # scale_matrix is false, if scale_matrix is true, just scale the matrix
        self.rescale_matrix(scale_matrix)

        # Check if covariance matrices are valid
        assert_valid_covariance(state_noise_cov)
        assert_valid_covariance(obs_noise_cov)

        # Check shapes
        if state_matrix.shape[0] != state_matrix.shape[1]:
            raise ValueError("State matrix is not square")
        if obs_matrix.shape[1] != self.d:
            raise ValueError("Obs matrix is not (p,d)")
        if state_noise_cov.shape[0] != self.d:
            raise ValueError("State cov matrix is not (d,d)")
        if obs_noise_cov.shape[0] != self.p:
            raise ValueError("Obs cov matrix is not (d,d)")

    def rescale_matrix(self, scale_matrix: bool) -> None:
        """
        Function that re-scales the state matrix to make it stable

        Parameter:
            scale_matrix: Whether to rescale the state matrix
        """
        eigenvalues = np.linalg.eigvals(self.state_matrix)
        spectral_radius = np.max(np.abs(eigenvalues))
        if spectral_radius >= 1:
            if scale_matrix:
                self.state_matrix /= (spectral_radius + 0.1)
            else:
                warnings.warn('Warning: state_matrix is not stable.')

    def copy_with_changes(self, **kwargs) -> Self:
        """
        Copy the initial parameters of this object into another object. Allow
        kwargs to change the initial values of the copy object.
        """

        # Copy current object
        other = copy.deepcopy(self)

        # Copy changed arguments
        for k in kwargs:
            other.__dict__[k] = kwargs[k]

        scale_key = "scale_matrix"
        if scale_key in kwargs:
            other.rescale_matrix(kwargs[scale_key])

        return other

    def evolve_state(self, state: NDArray) -> NDArray:
        """
        Docstring TODO
        """
        mv_n = self.rng.multivariate_normal
        draws = mv_n(np.zeros((self.d)), self.state_noise_cov,
                     size=(state.shape[0]))[:, :, np.newaxis]

        return self.state_matrix @ state + draws

    def get_obs(self, state: NDArray) -> NDArray:
        """
        Docstring TODO
        """
        mv_n = self.rng.multivariate_normal
        draws = mv_n(np.zeros((self.p)), self.obs_noise_cov,
                     size=(state.shape[0]))[:, :, np.newaxis]

        return self.obs_matrix @ state + draws

    def sample(self, n: int, t: int, init_mean: NDArray | None = None,
               init_cov: NDArray | None = None) -> NDArray:
        """
        Samples from LinearSSM with initial state sampled from Gaussian with
        init_mean and init_cov.
        Default: mean zero with unit variance.

        Parameters:
            n: number of time series replicates to sample
            t: number of time points in each time series
            init_mean: shape (d,1) mean values for initial state sampling
            init_cov: shape (d,1) covariance for initial state sampling

        Returns:
            Numpy array of shape (n,p,t) representing generated time series
        """
        self.ts_samples = np.zeros((n, self.p, t))
        if init_mean is None:
            init_mean = np.zeros((self.d))
        if init_cov is None:
            init_cov = np.eye(self.d)

        # Sample initial state
        state = self.rng.multivariate_normal(init_mean, init_cov,
                                             size=n)[:, :, np.newaxis]

        # Recursively sample observations
        for t_index in range(t):
            self.ts_samples[:, :, t_index] = self.get_obs(state)[:, :, 0]
            state = self.evolve_state(state)

        return self.ts_samples

    def plot_sample(self) -> None:
        """
        Plot the time series sample
        """
        plt.figure()

        for i in range(self.p):
            subplot_index = int(f"{self.p}1{i + 1}")
            plt.subplot(subplot_index)

            plt.plot(self.ts_samples[:, i, :].T)
            plt.title(f"Dim. {i}")

        plt.xlabel('Time')
        plt.show()


class HSMM(TimeSeries):

    def __init__(self, init_probs: NDArray, transition_probs: NDArray,
                 emission_means: list[NDArray],
                 emission_covariances: list[NDArray],
                 state_durations_params: list[tuple]) -> None:
        """
        Initialize the Hidden Semi-Markov Model with multivariate emissions.

        Parameters:
            init_probs: A 1D numpy array of initial state probabilities
                (must sum to 1).
            transition_probs: A 2D numpy array (N x N) where N is the number of
                states. Each entry represents the probability of transitioning
                from one state to another.
            emission_means: A list of mean vectors (1D numpy arrays) for
                emissions in each state.
            emission_covariances: A list of covariance matrices
                (2D numpy arrays) for emissions in each state.
            state_durations_params: A list of tuples
                [(mean, std, min, max), ...] representing the mean, standard
                deviation, and truncation range for the duration of each state.
        """
        super().__init__()

        self.init_probs = np.array(init_probs)
        self.transition_probs = np.array(transition_probs)
        self.emission_means = [np.array(mean) for mean in emission_means]
        self.emission_covariances = [np.array(cov) for cov in
                                     emission_covariances]
        self.state_durations_params = state_durations_params

        # Validate inputs
        num_states = len(self.init_probs)
        if not np.isclose(np.sum(self.init_probs), 1):
            raise ValueError("Initial state probabilities must sum to 1.")
        if self.transition_probs.shape[0] != self.transition_probs.shape[1]:
            raise ValueError("Transition matrix must be square.")
        if self.transition_probs.shape[0] != num_states:
            e = "Transition matrix dimensions must match the number of states."
            raise ValueError(e)
        if not np.allclose(self.transition_probs.sum(axis=1), 1):
            e = "Each row of the transition matrix must sum to 1."
            raise ValueError(e)
        if len(self.emission_means) != num_states:
            e = "Number of emission means must match the number of states."
            raise ValueError(e)
        if len(self.emission_covariances) != num_states:
            e = "Number of emission covariances must match the "
            e += "number of states."
            raise ValueError(e)
        if len(self.state_durations_params) != num_states:
            e = "Number of state duration parameters must match the "
            e += "number of states."
            raise ValueError(e)

        for cov in self.emission_covariances:
            if cov.shape[0] != cov.shape[1]:
                raise ValueError("Each covariance matrix must be square.")
            if not np.allclose(cov, cov.T):
                raise ValueError("Each covariance matrix must be symmetric.")
            if not np.all(np.linalg.eigvals(cov) >= 0):
                e = "Each covariance matrix must be positive semi-definite."
                raise ValueError(e)

    def truncated_discrete_normal(self, mean: float, std: float,
                                  min_val: float, max_val: float,
                                  size: int = 1) -> NDArray:
        """
        Sample from a truncated discrete normal distribution.

        Parameters:
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            min_val: Minimum truncation value.
            max_val: Maximum truncation value.
            size: Number of samples to generate.

        Returns:
            A NumPy array of samples.
        """
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        samples = truncnorm(a, b, loc=mean, scale=std).rvs(size=size)

        return np.clip(np.round(samples), min_val, max_val).astype(int)

    def sample(self, n: int, t: int) -> tuple[list[NDArray], list[NDArray]]:
        """
        Generate n time series of length t from the Hidden Semi-Markov Model
        with multivariate emissions.

        Parameters:
            n: Number of time series to generate.
            t: Length of each time series.

        Returns:
            samples: A list of n numpy arrays, each of shape (t, D),
                representing the multivariate time series.
            states: A list of n numpy arrays, each of shape (t,), representing
                the hidden state sequence.
        """
        num_states = len(self.emission_means)
        samples = []
        states = []

        for _ in range(n):
            sequence: list[NDArray] = []
            state_sequence = []

            # Start in a random initial state
            current_state = np.random.choice(num_states, p=self.init_probs)
            time = 0

            while time < t:
                # Sample duration from truncated discrete normal distribution
                values = self.state_durations_params[current_state]
                mean, std, min_val, max_val = values
                duration = self.truncated_discrete_normal(mean, std, min_val,
                                                          max_val, size=1)[0]

                # Limit duration to avoid exceeding the time series length
                duration = min(duration, t - time)

                # Generate multivariate emissions for the duration
                em = self.emission_means[current_state]
                ec = self.emission_covariances[current_state]
                emissions = np.random.multivariate_normal(em, ec, duration)
                sequence.extend(emissions)
                state_sequence.extend([current_state] * duration)

                # Transition to the next state
                tp = self.transition_probs[current_state]
                next_state = np.random.choice(num_states, p=tp)
                current_state = next_state

                time += duration

            # Append the generated sequence and states
            samples.append(np.array(sequence))
            states.append(np.array(state_sequence))

        return samples, states


class Copula(TimeSeries):

    def __init__(self, alpha: float | None = None,
                 markovian: bool = True) -> None:
        """
        Initializes a new instance of a Copula.

        Parameters:
            alpha: parameter controlling the copula, currently only
                archimedian copulas are implemented
            markovian: default True, if True, then the copula samples have a
                markovian process property, i.e. correlated in time.
        The functions here-in rely heavily on Chapter 2 of the book:
            Sun, Li-Hsien, et al. Copula-Based Markov Models for Time Series:
            Parametric Inference and Process Control,
            Springer Singapore Pte. Limited, 2020.
        """
        super().__init__()
        self.alpha = alpha
        self.markovian = markovian

        # Family that will be defined by instance of subclass
        self.copula_family: str | None = None

    def define_marginal(self, marginal_family: str = "uniform",
                        loc: float | None = None,
                        scale: float | None = None) -> None:
        """
        Method to instantiate the desired marginal distribution qualities.

        Arguments:
            marginal_family: string, choices include:
                "uniform", "normal", "gamma", "t", "gumbel", "exponential"
            loc: location parameter for distributions that accept them for
                exponential, loc corresponds to rate for t, loc corresponds
                to degrees of freedom
            scale: scale parameter for distributions that accept them
        """
        self.marginal_family = marginal_family
        self.loc = loc
        self.scale = scale

    def variable_generator(self, u: NDArray, w: NDArray) -> NDArray:
        """
        The variable_generator used for the copula model to produce
        realizations from its PDF.

        Parameters:
            u: number between (0,1)
            w: number between (0,1)
        Returns v such that (U,V) is a realization of the copula model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def variable(self, u: NDArray | None = None, w: NDArray | None = None,
                 *args) -> float | NDArray:
        """
        Provides variables u and w to the copula variable_generator and returns
        the variable v such that (U,V) is a realization of the copula model.

        If u and w are random or None, then provides a random variable.

        Parameters:
            u: uniform variable between (0,1)
            w: uniform variable between (0,1)
        """
        if u is None:
            u = stats.uniform.rvs(0, 1)
        if w is None:
            w = stats.uniform.rvs(0, 1)

        return self.variable_generator(u, w)

    def sample(self, n: int = 1, t: int = 1000) -> NDArray:
        """
        Generate n samples from a bivariate copula distribution with marginal
        family specified.

        If markovian, x_t and x_t+1 will be distributed according to the
        copula. The marginal distribution will be uniform.

        Parameters:
            n: number of samples to draw of length t
            t: number of time points
        Returns ndarray of random variables of shape [n, t]
        """
        # Initialize empty matrix
        uniform_samples = np.zeros((n, t))

        for nn in range(n):
            # Random initialization (get the chain going)
            u = stats.uniform.rvs(0, 1)
            for tt in range(t):
                # Do not provide w to get a random sample
                v = self.variable(u=u, w=None)
                uniform_samples[nn, tt] = v

                if self.markovian:
                    u = v

        self.uniform_samples = uniform_samples
        self.samples = self.uniform_to_marginal(self.uniform_samples)

        return self.samples

    def density_generator(self, u: NDArray, v: NDArray) -> NDArray:
        """
        The joint density of the bivariate copula

        Arguments:
            u: value between (0,1)
            v: value between (0,1)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def density(self, u: NDArray | None = None, v: NDArray | None = None,
                n_samples: int = 2000) -> tuple[NDArray, NDArray, NDArray]:
        """
        Function to call the density_generator function specific to each
        copula. If no arguments are given, then provides a meshgrid that can
        easily be passed to plt.contour()

        Arguments:
            u: array
            v: array of same size as u
            n_samples: number of samples to draw if neither u or v are given
            Must provide both u and v or neither u and v.
        Returns u, v, p with p being the density values
        """

        if u is None and v is None:
            u_vec = np.linspace(0.001, 0.999, n_samples)
            v_vec = np.linspace(0.001, 0.999, n_samples)
            u, v = np.meshgrid(u_vec, v_vec)

        if u is None or v is None:
            e = "Must provide both u and v or neither u and v."
            raise ValueError(e)

        p = self.density_generator(u, v)

        return u, v, p

    def uniform_to_marginal(self, uniform_samples: NDArray) -> NDArray:
        """
        Convert the data to the uniform marginals to have a marginal
        distribution of interest.

        Generates random variables from uniform RVs using inverse CDF from
        distribution specified by family. Updates class with valid loc and
        scale parameters if None was provided.

        Arguments:
            uniform_samples: array of samples between (0,1)
        Returns samples after being passed through the marginal distribution.
        """
        if self.marginal_family == "uniform":
            x = uniform_samples  # do nothing, keep marginal uniform

        elif self.marginal_family == "normal":
            if self.loc is None:
                self.loc = 0
            if self.scale is None:
                self.scale = 1
            x = stats.norm.ppf(uniform_samples, self.loc, self.scale)

        elif self.marginal_family == "gamma":
            if self.loc is None:
                self.loc = 1
            if self.scale is None:
                self.scale = 2
            x = stats.gamma.ppf(uniform_samples, self.loc, self.scale)

        elif self.marginal_family == "t":
            if self.loc is None:
                self.loc = 3  # really degrees of freedom
            x = stats.t.ppf(uniform_samples, self.loc)

        elif self.marginal_family == "gumbel":
            if self.loc is None:
                self.loc = 3
            if self.scale is None:
                self.scale = 4
            x = stats.gumbel_r.ppf(uniform_samples, self.loc, self.scale)

        elif self.marginal_family == "exponential":
            if self.loc is None:
                self.loc = 1  # rate
            x = stats.expon.ppf(uniform_samples, self.loc)

        else:
            e = "Marginal family chosen is not supported. "
            e += "Please see possible options."
            raise ValueError(e)

        return x


class ClaytonCopula(Copula):
    """
    Clayton: lower tail dependence (strong dependence at (U, V)≈(0,0)). The
    level of dependence increases as α increases. bounds: [-1, inf). This code
    does not support alpha = -1

    "The Clayton and Frank copulas converge to the independence copula
    when α → 0."

    "The Clayton copula allows negative dependence for α ∈ [−1, 0),
    particularly giving the Fréchet–Hoeffding lower bound copula for α = −1.

    However, the copula density does not exist on the line u−α+v−α−1 = 0,
    giving a “singular” distribution for α ∈ [−1, 0).

    Hence, the Clayton copula is usually not suitable for modeling negative
    dependence except for some special examples (e.g., Emura et al. 2011)."
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.copula_family = "Clayton"

        assert self.alpha is not None
        if (self.alpha <= -1) | (self.alpha == 0):
            e = "Parameter bounds may be [-1, inf) and not 0 for Clayton "
            e += "copula. This code does not support alpha = -1"
            raise ValueError(e)

    def variable_generator(self, u: NDArray, w: NDArray) -> NDArray:

        assert self.alpha is not None
        v = ((w ** (-self.alpha / (self.alpha + 1)) - 1)
             * u ** (-self.alpha) + 1) ** (-1 / self.alpha)

        return v

    def density_generator(self, u: NDArray, v: NDArray) -> NDArray:
        # Bivariate density
        assert self.alpha is not None
        p = (1 + self.alpha) * (u * v) ** (-self.alpha - 1)
        p *= (u ** -self.alpha + v ** -self.alpha - 1) ** (-1/self.alpha - 2)

        return p


class JoeCopula(Copula):
    """
    Joe:  upper tail dependence (strong dependence at (U, V)≈(1,1)). The level
    of dependence increases as α increases.

    "The Joe copula becomes the independent copula when α = 1." - Copula book
    Getting these variables requires a somewhat finicky solver/optimization.
    bounds: [1, inf).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.copula_family = "Joe"

        assert self.alpha is not None
        if self.alpha < 1:
            e = "Parameter bounds may be [1, inf) for Joe copula"
            raise ValueError(e)
        if self.alpha > 200:
            w = "Large values of alpha (>200) can cause instability in the "
            w += "number generation and it is recommended to keep alpha below "
            w += "this value."
            warnings.warn(w)

    def solver(self, v: NDArray, u: NDArray, w: NDArray) -> NDArray:
        """
        Solver function for the joe copula.
        Parameters:
            v: array of places to initialize solver
            u: uniform variable
            w: uniform variable
        """

        assert self.alpha is not None
        term1 = (1 - u) ** self.alpha + (1 - v) ** self.alpha
        term1 -= ((1 - u) * (1 - v)) ** self.alpha
        term1 **= 1 / self.alpha - 1

        term2 = (1 - (1 - v) ** self.alpha) * (1 - u) ** (self.alpha - 1)

        return term1 * term2 - w

    def variable_generator(self, u: NDArray, w: NDArray) -> NDArray:

        # First, get a good spot for initializing the solver
        ntry = 1000
        vtry = np.linspace(0.01, 0.99, ntry)
        min_index = np.argmin(self.solver(vtry, u, w) ** 2)

        # Initialization
        v0 = vtry[min_index]
        roots = fsolve(self.solver, v0, (u, w))

        if len(roots) > 1:
            print("Multiple roots, troubleshoot solver")

        return roots[0]

    def density_generator(self, u: NDArray, v: NDArray) -> NDArray:

        # Precalculate common factors
        assert self.alpha is not None
        product = (1 - u) * (1 - v)
        common = (1 - u) ** self.alpha + (1 - v) ** self.alpha
        common -= product ** self.alpha

        term1 = self.alpha * common ** (1 / self.alpha - 1)
        term1 *= product ** (self.alpha - 1)

        term2 = (self.alpha - 1) * common ** (1 / self.alpha - 2)

        term3 = (1 - (1 - u) ** self.alpha) * (1 - (1 - v) ** self.alpha)
        term3 *= product ** (self.alpha - 1)

        # Bivariate density
        return term1 + term2 * term3


class FrankCopula(Copula):
    """
    Frank: A negative value of α gives negative dependence while a positive
    value of α gives positive dependence between U and V.

    Getting these variables requires a somewhat finicky solver/optimization.
    bounds: [-inf, 0) or (0, inf].

    The Clayton and Frank copulas converge to the independence
    copula when α → 0.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.copula_family = "Frank"

        assert self.alpha is not None
        if self.alpha == 0:
            e = "Parameter bounds may be (-inf, inf) but not 0 for "
            e += "Frank copula"
            raise ValueError(e)
        if self.alpha > 500:
            w = "Large values of alpha (>500) can cause instability in the "
            w += "number generation and it is recommended to keep alpha "
            w += "below 500."
            warnings.warn(w)

    def variable_generator(self, u: NDArray, w: NDArray) -> NDArray:

        assert self.alpha is not None
        term1 = -1 / self.alpha
        lognum = np.exp(-self.alpha * u) - w * \
            np.exp(-self.alpha * u) + w * np.exp(-self.alpha)

        if lognum == 0:
            print(u, w, self.alpha)

        logden = np.exp(-self.alpha * u) - w * np.exp(-self.alpha * u) + w

        return term1 * (np.log(lognum) - np.log(logden))

    def density_generator(self, u: NDArray, v: NDArray) -> NDArray:

        assert self.alpha is not None
        num = self.alpha * (1 - np.exp(-self.alpha)) * \
            np.exp(-self.alpha * (u + v))
        den = (np.exp(-self.alpha) - 1 + (np.exp(-self.alpha * u) - 1)
               * (np.exp(-self.alpha * v) - 1))**2

        # Bivariate density
        return num / den


class NormalCopula(Copula):
    """
    Normal: the parameter specified controls the correlation, bounds: (-1,1)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.copula_family = "Normal"

        assert self.alpha is not None
        if np.abs(self.alpha) > 1:
            e = "Parameter bounds may be (-1,1) for Normal copula"
            raise ValueError(e)

    def variable_generator(self, u: NDArray, w: NDArray) -> NDArray:
        psi_i_k = stats.norm.ppf(w)
        psi_i_u = stats.norm.ppf(u)

        assert self.alpha is not None
        inner_term = psi_i_k * \
            np.sqrt(1 - self.alpha ** 2) + self.alpha * psi_i_u
        v = stats.norm.cdf(inner_term)
        return v

    def density_generator(self, u: NDArray, v: NDArray) -> NDArray:
        psi_i_u = stats.norm.ppf(u)
        psi_i_v = stats.norm.ppf(v)

        assert self.alpha is not None
        term1 = 1 / np.sqrt(1 - self.alpha**2)

        # Note, in the following term, the book has a sign wrong, this has
        # been corrected here:
        exp_num = - self.alpha**2 * \
            (psi_i_u**2 + psi_i_v**2) + 2 * self.alpha * psi_i_u * psi_i_v
        exp_den = 2 * (1 - self.alpha**2)

        # Bivariate density
        return term1 * np.exp(exp_num / exp_den)
