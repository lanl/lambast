"""
Time series generation classes.

"""

import warnings
import numpy as np

from util import is_valid_covariance


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
        ts_samples[:, :, 0] = self.get_observation(state).squeeze()
        # recursively sample observations
        for t_index in range(1, t):
            state = self.evolve_state(state)
            ts_samples[:, :, t_index] = self.get_observation(state).squeeze()
        return ts_samples


class HiddenSemiMarkovModel(TimeSeries):

    def __init__(self, init_probs, transition_probs, emission_means,
                 emission_covariances, state_durations):
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
            state_durations: A list of lists or arrays, where each sublist
                represents the probability distribution of durations
                for each state.
        """
        super().__init__()

        self.init_probs = np.array(init_probs)
        self.transition_probs = np.array(transition_probs)
        self.emission_means = [np.array(mean) for mean in emission_means]
        self.emission_covariances = [np.array(cov) for cov in
                                     emission_covariances]
        self.state_durations = [np.array(durations) for durations in
                                state_durations]

        # Validate inputs
        if not np.isclose(np.sum(self.init_probs), 1):
            raise ValueError("Initial state probabilities must sum to 1.")
        if self.transition_probs.shape[0] != self.transition_probs.shape[1]:
            raise ValueError("Transition matrix must be square.")
        if not np.allclose(self.transition_probs.sum(axis=1), 1):
            e = "Each row of the transition matrix must sum to 1."
            raise ValueError(e)
        if len(self.emission_means) != len(self.emission_covariances) or \
                len(self.emission_means) != len(self.state_durations):
            e = "Mismatch between the number of states and the"
            e += " provided parameters"
            raise ValueError(e)

        for cov in self.emission_covariances:
            if cov.shape[0] != cov.shape[1]:
                raise ValueError("Each covariance matrix must be square.")
            if not np.allclose(cov, cov.T):
                raise ValueError("Each covariance matrix must be symmetric.")
            if not np.all(np.linalg.eigvals(cov) >= 0):
                e = "Each covariance matrix must be positive semi-definite."
                raise ValueError(e)

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
                # Sample a duration from the current state's
                # duration distribution
                sd = self.state_durations[current_state]
                duration = np.random.choice(len(sd), p=sd)

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


# Add some driver code as a test for the above classes
if __name__ == "__main__":

    import matplotlib.pyplot as plt

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
        # TODO
        raise NotImplementedError("Driver not implemented")

    driver_LinearSSM()
