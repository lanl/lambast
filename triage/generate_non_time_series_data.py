"""
Classes to generate non-time series data.
"""

import numpy as np


class BaseDataClass:

    def __init__(self):
        """
        Base class for non-timeseries data.
        """

    def sample(self, num_samples, *args, **kwargs):
        """
        Sample `n` points from the BaseDataClass model.

        Parameters:
        - num_samples (int): number of points to sample
        """
        raise NotImplementedError("Subclasses must implement this method.")


class MultivariateGaussian(BaseDataClass):
    def __init__(self, mean, covariance):
        """
        Initialize the Multivariate Gaussian model.

        Parameters:
        - mean: A 1D numpy array representing the mean vector.
        - covariance: A 2D numpy array representing the covariance matrix.
        """
        super().__init__()

        self.mean = np.array(mean)
        self.covariance = np.array(covariance)

        # Validate covariance matrix
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self.mean.shape[0] != self.covariance.shape[0]:
            e = "Mean vector size must match covariance matrix dimensions."
            raise ValueError(e)
        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("Covariance matrix must be symmetric.")
        if not np.all(np.linalg.eigvals(self.covariance) >= 0):
            e = "Covariance matrix must be positive semi-definite."
            raise ValueError(e)

    def sample(self, num_samples=1):
        """
        Generate samples from the multivariate Gaussian distribution.

        Parameters:
        - num_samples: Number of samples to generate.

        Returns:
        - samples: A 2D numpy array where each row is a sample.
        """
        samples = np.random.multivariate_normal(self.mean, self.covariance,
                                                num_samples)
        return samples


class GaussianMixtureModel(BaseDataClass):
    def __init__(self, weights, means, covariances):
        """
        Initialize the Gaussian Mixture Model.

        Parameters:
        - weights: A list or 1D numpy array of weights for each component
            (should sum to 1).
        - means: A list of 1D numpy arrays representing the mean vector for
            each component.
        - covariances: A list of 2D numpy arrays representing the covariance
            matrix for each component.
        """
        super().__init__()

        self.weights = np.array(weights)
        self.means = [np.array(mean) for mean in means]
        self.covariances = [np.array(cov) for cov in covariances]

        # Validate inputs
        if not np.isclose(np.sum(self.weights), 1):
            raise ValueError("Weights must sum to 1.")
        if len(self.weights) != len(self.means) or \
                len(self.means) != len(self.covariances):

            e = "Number of weights, means, and covariances must be the same."
            raise ValueError(e)

        for mean, cov in zip(self.means, self.covariances):
            # Check if the covariance matrix is square
            if cov.shape[0] != cov.shape[1]:
                raise ValueError("Each covariance matrix must be square.")
            # Check if the dimensions of the mean match the covariance matrix
            if len(mean) != cov.shape[0]:
                e = "The size of the mean vector must match the dimensions of"
                e += " the covariance matrix."
                raise ValueError(e)
            # Check if the covariance matrix is symmetric
            if not np.allclose(cov, cov.T):
                raise ValueError("Each covariance matrix must be symmetric.")
            # Check if the covariance matrix is positive semi-definite
            if not np.all(np.linalg.eigvals(cov) >= 0):
                e = "Each covariance matrix must be positive semi-definite."
                raise ValueError(e)

    def sample(self, num_samples=1):
        """
        Generate samples from the Gaussian Mixture Model.

        Parameters:
        - num_samples: Number of samples to generate.

        Returns:
        - samples: A 2D numpy array where each row is a sample.
        - labels: A 1D numpy array of component labels corresponding to the
            samples.
        """
        n_components = len(self.weights)
        samples = []
        labels = []

        # Generate samples
        for _ in range(num_samples):
            # Select a component based on the weights
            component = np.random.choice(n_components, p=self.weights)
            # Sample from the selected Gaussian component
            sample = np.random.multivariate_normal(self.means[component],
                                                   self.covariances[component])
            samples.append(sample)
            labels.append(component)

        return np.array(samples), np.array(labels)
