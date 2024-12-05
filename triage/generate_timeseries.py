"""
Time series generation classes.

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from .util import is_valid_covariance

class TimeSeries:

    def __init__(self):
        """
        TimeSeries base class.

        """

    def sample(self, n, t, *args, **kwargs):
        """
        Sample `n` time series with `t` time points from the TimeSeries model.

        Parameters:
            n (int): number of time series replicates to sample
            t (int): number of time points in each time series
        """
        raise NotImplementedError("Subclasses must implement this method.")


class LinearSSM(TimeSeries):

    def __init__(self, state_matrix, state_noise_cov, obs_matrix,
                 obs_noise_cov, rng=None):
        """
        Initializes a new instance of a Linear State Space Model.
        State dimension is d, observation dimension is p.
        Assume zero-mean Gaussian noise in state and observation space.

        Parameters:
            state_matrix (np.ndarray): shape (d,d) numpy array defining state
                transition matrix
            state_noise_cov (np.ndarray): shape (d,d) numpy array defining
                state noise covariance
            obs_matrix (np.ndarray): shape (p,d) numpy array defining
                observation matrix
            obs_noise_cov (np.ndarray): shape (p,p) numpy array defininig
                observation noise covariance
            rng: numpy rng (else use default); see
                https://numpy.org/doc/2.0/reference/random/index.html#random-quick-start
        """
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.d = len(state_matrix)
        self.p = len(obs_matrix)

        # check inputs
        # check if state matrix is stable and warn user if not
        eigenvalues = np.linalg.eigvals(state_matrix)
        spectral_radius = max(abs(eigenvalues))
        eigenvalues = np.linalg.eigvals(state_matrix)
        if spectral_radius >= 1:
            warnings.warn('Warning: state_matrix is not stable.')
        # check if covariance matrices are valid
        is_valid_covariance(state_noise_cov)
        is_valid_covariance(obs_noise_cov)
        # check shapes
        if state_matrix.shape[0] != state_matrix.shape[1]:
            raise ValueError("State matrix is not square")
        if obs_matrix.shape[1] != self.d:
            raise ValueError("Obs matrix is not (p,d)")
        if state_noise_cov.shape[0] != self.d:
            raise ValueError("State cov matrix is not (d,d)")
        if obs_noise_cov.shape[0] != self.p:
            raise ValueError("Obs cov matrix is not (d,d)")
        self.state_matrix = state_matrix
        self.state_noise_cov = state_noise_cov
        self.obs_matrix = obs_matrix
        self.obs_noise_cov = obs_noise_cov

    def evolve_state(self, state):
        """
        Docstring TODO
        """
        mv_n = self.rng.multivariate_normal
        draws = mv_n(np.zeros((self.d)), self.state_noise_cov,
                     size=(len(state)))[:, :, np.newaxis]

        return self.state_matrix @ state + draws

    def get_observation(self, state):
        """
        Docstring TODO
        """
        mv_n = self.rng.multivariate_normal
        draws = mv_n(np.zeros((self.p)), self.obs_noise_cov,
                     size=(len(state)))[:, :, np.newaxis]

        return self.obs_matrix @ state + draws

    def sample(self, n, t, init_mean=None, init_cov=None):
        """
        Samples from LinearSSM with initial state sampled from Gaussian with
        init_mean and init_cov.
        Default: mean zero with unit variance.

        Parameters:
            n (int): number of time series replicates to sample
            t (int): number of time points in each time series
            init_mean (np.ndarray): shape (d,1) mean values for initial state
                sampling
            init_cov (np.ndarray): shape (d,1) covariance for initial state
                sampling

        Returns:
            Numpy array of shape (n,p,t) representing generated time series
        """
        ts_samples = np.zeros((n, self.p, t))
        if init_mean is None:
            init_mean = np.zeros((self.d))
        if init_cov is None:
            init_cov = np.eye(self.d)
        # sample initial state
        state = self.rng.multivariate_normal(init_mean, init_cov,
                                             size=n)[:, :, np.newaxis]
        # sample initial observation
        ts_samples[:, :, 0] = self.get_observation(state)[:, :, 0]
        # recursively sample observations
        for t_index in range(1, t):
            state = self.evolve_state(state)
            ts_samples[:, :, t_index] = self.get_observation(state)[:, :, 0]
        return ts_samples


class HiddenSemiMarkovModel(TimeSeries):

    def __init__(self, init_probs, transition_probs, emission_means,
                 emission_covariances, state_durations_params):
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

    def truncated_discrete_normal(self, mean, std, min_val, max_val, size=1):
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
        a, b = (min_val - mean) / std, (max_val - mean) / std
        samples = truncnorm(a, b, loc=mean, scale=std).rvs(size=size)
        return np.clip(np.round(samples), min_val, max_val).astype(int)

    def sample(self, n, t):
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
            sequence = []
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

    def __init__(self, alpha=None, markovian=True):
        """
        Initializes a new instance of a Copula.

        Parameters:
            alpha (np.ndarray): parameter controlling the copula, currently
                only archimedian copulas are implemented
            markovian: boolean, default True, if True, then the
                copula samples have a markovian process property, i.e.
                correlated in time.
        The functions here-in rely heavily on Chapter 2 of the book:
            Sun, Li-Hsien, et al. Copula-Based Markov Models for Time Series:
            Parametric Inference and Process Control,
            Springer Singapore Pte. Limited, 2020.
        """
        super().__init__()
        self.alpha = alpha
        self.markovian = markovian

        # Family that will be defined by instance of subclass
        self.copula_family = None

    def define_marginal(self, marginal_family="uniform", loc=None, scale=None):
        """
        Method to instantiate the desired marginal distribution qualities.

        Arguments:
            marginal_family: string, choices include:
                "uniform", "normal", "gamma", "t", "gumbel", "exponential"
            loc: float64 (any number), location parameter for distributions
                that accept them
                for exponential, loc corresponds to rate
                for t, loc corresponds to degrees of freedom
            scale: float64 (any number), scale parameter for distributions
                that accept them
        """
        self.marginal_family = marginal_family
        self.loc = loc
        self.scale = scale

    def variable_generator(self, u, w):
        """
        The variable_generator used for the copula model to produce
        realizations from its PDF.

        Parameters:
            u: number between (0,1)
            w: number between (0,1)
        Returns v such that (U,V) is a realization of the copula model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def variable(self, u=None, w=None, *args):
        """
        Provides variables u and w to the copula variable_generator and returns
        the variable v such that (U,V) is a realization of the copula model.

        If u and w are random or None, then provides a random variable.

        Parameters:
            u: uniform variable between (0,1)
            w: uniform variable between (0,1)
        """
        from scipy import stats
        if u is None:
            u = stats.uniform.rvs(0, 1)
        if w is None:
            w = stats.uniform.rvs(0, 1)

        return self.variable_generator(u, w)

    def sample(self, n=1, t=1000):
        """
        Generate n samples from a bivariate copula distribution with marginal
        family specified.

        If markovian, x_t and x_t+1 will be distributed according to the
        copula. The marginal distribution will be uniform.

        Parameters:
            n: integer, default 2, number of samples to draw of length t
            t: integer, default 1000, number of time points
        Returns ndarray of random variables of shape [n, t]
        """
        from scipy import stats
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

    def density_generator(self, u, v):
        """
        The joint density of the bivariate copula

        Arguments:
            u: value between (0,1)
            v: value between (0,1)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def density(self, u=None, v=None, n_samples=2000):
        """
        Function to call the density_generator function specific to each
        copula. If no arguments are given, then provides a meshgrid that can
        easily be passed to plt.contour()

        Arguments:
            u: np.array()
            v: np.array() of same size as u
            n_samples: int, number of samples to draw if neither u or v are
                given
            Must provide both u and v or neither u and v.
        Returns u, v, p with p being the density values
        """
        if (u is None) and (v is None):
            u_vec = np.linspace(0.001, 0.999, n_samples)
            v_vec = np.linspace(0.001, 0.999, n_samples)
            u, v = np.meshgrid(u_vec, v_vec)
        elif None in [u, v]:
            e = "Must provide both u and v or neither u and v."
            raise ValueError(e)

        p = self.density_generator(u, v)

        return u, v, p

    def uniform_to_marginal(self, uniform_samples):
        """
        Convert the data to the uniform marginals to have a marginal
        distribution of interest.

        Generates random variables from uniform RVs using inverse CDF from
        distribution specified by family. Updates class with valid loc and
        scale parameters if None was provided.

        Arguments:
            uniform_samples: np.array() of samples between (0,1)
        Returns samples after being passed through the marginal distribution.
        """
        from scipy import stats
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


class Clayton(Copula):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copula_family = "Clayton"
        if (self.alpha <= -1) | (self.alpha == 0):
            e = "Parameter bounds may be [-1, inf) and not 0 for Clayton "
            e += "copula. This code does not support alpha = -1"
            raise ValueError(e)

    def variable_generator(self, u, w):
        v = ((w ** (-self.alpha / (self.alpha + 1)) - 1)
             * u ** (-self.alpha) + 1) ** (-1 / self.alpha)

        return v

    def density_generator(self, u, v):
        # Bivariate density
        p = (1 + self.alpha) * (u * v) ** (-self.alpha - 1)
        p *= (u ** -self.alpha + v ** -self.alpha - 1) ** (-1/self.alpha - 2)

        return p


class Joe(Copula):
    """
    Joe:  upper tail dependence (strong dependence at (U, V)≈(1,1)). The level
    of dependence increases as α increases.

    "The Joe copula becomes the independent copula when α = 1." - Copula book
    Getting these variables requires a somewhat finicky solver/optimization.
    bounds: [1, inf).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copula_family = "Joe"
        if self.alpha < 1:
            e = "Parameter bounds may be [1, inf) for Joe copula"
            raise ValueError(e)
        if self.alpha > 200:
            w = "Large values of alpha (>200) can cause instability in the "
            w += "number generation and it is recommended to keep alpha below "
            w += "this value."
            warnings.warn(w)

    def solver(self, v, u, w):
        """
        Solver function for the joe copula.
        Parameters:
            v: np.ndarray of places to initialize solver
            u: uniform variable
            w: uniform variable
        """

        term1 = (1 - u) ** self.alpha + (1 - v) ** self.alpha
        term1 -= ((1 - u) * (1 - v)) ** self.alpha
        term1 **= 1 / self.alpha - 1

        term2 = (1 - (1 - v) ** self.alpha) * (1 - u) ** (self.alpha - 1)

        return term1 * term2 - w

    def variable_generator(self, u, w):
        from scipy.optimize import fsolve

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

    def density_generator(self, u, v):

        # Precalculate common factors
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


class Frank(Copula):
    """
    Frank: A negative value of α gives negative dependence while a positive
    value of α gives positive dependence between U and V.

    Getting these variables requires a somewhat finicky solver/optimization.
    bounds: [-inf, 0) or (0, inf].

    The Clayton and Frank copulas converge to the independence
    copula when α → 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copula_family = "Frank"
        if self.alpha == 0:
            e = "Parameter bounds may be (-inf, inf) but not 0 for "
            e += "Frank copula"
            raise ValueError(e)
        if self.alpha > 500:
            w = "Large values of alpha (>500) can cause instability in the "
            w += "number generation and it is recommended to keep alpha "
            w += "below 500."
            warnings.warn(w)

    def variable_generator(self, u, w):
        term1 = -1 / self.alpha
        lognum = np.exp(-self.alpha * u) - w * \
            np.exp(-self.alpha * u) + w * np.exp(-self.alpha)

        if lognum == 0:
            print(u, w, self.alpha)

        logden = np.exp(-self.alpha * u) - w * np.exp(-self.alpha * u) + w

        return term1 * (np.log(lognum) - np.log(logden))

    def density_generator(self, u, v):

        num = self.alpha * (1 - np.exp(-self.alpha)) * \
            np.exp(-self.alpha * (u + v))
        den = (np.exp(-self.alpha) - 1 + (np.exp(-self.alpha * u) - 1)
               * (np.exp(-self.alpha * v) - 1))**2

        # Bivariate density
        return num / den


class Normal(Copula):
    """
    Normal: the parameter specified controls the correlation, bounds: (-1,1)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super().__init__()
        self.copula_family = "Normal"
        if np.abs(self.alpha) > 1:
            e = "Parameter bounds may be (-1,1) for Normal copula"
            raise ValueError(e)

    def variable_generator(self, u, w):
        from scipy import stats
        psi_i_k = stats.norm.ppf(w)
        psi_i_u = stats.norm.ppf(u)
        inner_term = psi_i_k * \
            np.sqrt(1 - self.alpha ** 2) + self.alpha * psi_i_u
        v = stats.norm.cdf(inner_term)
        return v

    def density_generator(self, u, v):
        from scipy import stats
        psi_i_u = stats.norm.ppf(u)
        psi_i_v = stats.norm.ppf(v)

        term1 = 1 / np.sqrt(1 - self.alpha**2)

        # Note, in the following term, the book has a sign wrong, this has
        # been corrected here:
        exp_num = - self.alpha**2 * \
            (psi_i_u**2 + psi_i_v**2) + 2 * self.alpha * psi_i_u * psi_i_v
        exp_den = 2 * (1 - self.alpha**2)

        # Bivariate density
        return term1 * np.exp(exp_num / exp_den)


# Add some driver code as a test for the above classes
if __name__ == "__main__":

    def driver_LinearSSM():
        """
        Driver test for the LinearSSM class
        """
        # small test case for now; probably need to put this somewhere else
        rng = np.random.default_rng(seed=42)
        d = 7
        p = 2
        n = 5
        t = 150
        # generate stable state matrix
        state_matrix = rng.normal(size=(d, d))
        eigenvalues = np.linalg.eigvals(state_matrix)
        spectral_radius = max(abs(eigenvalues))
        # Scale to ensure spectral radius < 1
        if spectral_radius >= 1:
            state_matrix = state_matrix / (spectral_radius + 0.1)
        obs_matrix = rng.normal(size=(p, d))
        state_noise_cov = 0.1 * np.eye(d)
        obs_noise_cov = 0.01 * np.eye(p)
        ssm = LinearSSM(state_matrix, state_noise_cov, obs_matrix,
                        obs_noise_cov, rng=rng)
        ts = ssm.sample(n, t)
        plt.figure()
        plt.subplot(211)
        plt.plot(ts[:, 0, :].T)
        plt.title("Dim. 0")
        plt.subplot(212)
        plt.plot(ts[:, 1, :].T)
        plt.title("Dim. 1")
        plt.xlabel('Time')
        plt.show()

    def driver_HiddenSemiMarkovModel():
        """
        Driver test for the LinearSSM class
        """
        # Define number of time series to generate and their length
        n, t = 5, 100  # Generate 5 time series of length 100

        # Define HSMM parameters

        # Initial state probabilities
        init_probs = [0.7, 0.3]

        transition_probs = [[0.8, 0.2], [0.3, 0.7]]  # Transition probabilities

        emission_means = [[0, 0], [3, 3]]  # Mean vectors for states 0 and 1

        # Covariance matrices for state 0 and 1
        emission_covariances = [[[1, 0.2], [0.2, 1]],
                                [[1, -0.3], [-0.3, 1]]]

        # Duration parameters for each state
        state_durations_params = [
            (10, 3, 1, t),  # State 1: mean=10, std=3, min=5, max=t
            (20, 5, 1, t),  # State 2: mean=20, std=5, min=10, max=t
        ]

        # Initialize the HSMM
        hsmm = HiddenSemiMarkovModel(init_probs, transition_probs,
                                     emission_means, emission_covariances,
                                     state_durations_params)

        # Generate time series data using above HSMM parameters
        samples, states = hsmm.sample(n, t)

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        # Define a color palette for the time series
        colors = plt.cm.viridis(np.linspace(0, 1, n))

        def do_plot(all_data, title, xlabel="Time",
                    ylabel="Value", label="Time Series"):
            """
            Factor out the plots
            """
            for i, (data, color) in enumerate(zip(all_data, colors)):
                axes[ax_index].plot(data, label=f"{label} {i + 1}",
                                    color=color)
            axes[ax_index].set_title(title)
            axes[ax_index].set_xlabel(xlabel)
            axes[ax_index].set_ylabel(ylabel)
            axes[ax_index].legend()

        # Transpose samples to access them easier for plot
        t_samples = np.transpose(samples)

        # Plot dimension 1 for all time series
        ax_index = 0
        do_plot(t_samples[0].T, "Dimension 1 of Time Series",
                label="Time Series")

        # Plot dimension 2 for all time series
        ax_index += 1
        do_plot(t_samples[1].T, "Dimension 2 of Time Series",
                label="Time Series")

        # Plot state sequences for all time series
        ax_index += 1
        do_plot(states, "State Sequences", ylabel="State",
                label="State Sequence")

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    driver_LinearSSM()
    driver_HiddenSemiMarkovModel()
