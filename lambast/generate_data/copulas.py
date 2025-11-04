import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import fsolve

from .timeseries_class import TimeSeries


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
                 rng: np.random.Generator | None = None,
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
            u = stats.uniform.rvs(0, 1, random_state=rng)
        if w is None:
            w = stats.uniform.rvs(0, 1, random_state=rng)

        return self.variable_generator(u, w)

    def sample(self, n: int = 1, t: int = 1000,
               rng: np.random.Generator | None = None) -> NDArray:
        """
        Generate n samples from a bivariate copula distribution with marginal
        family specified.

        If markovian, x_t and x_t+1 will be distributed according to the
        copula. The marginal distribution will be uniform.

        Parameters:
            n: number of samples to draw of length t
            t: number of time points
            rng: optional, the random number generator
        Returns ndarray of random variables of shape [n, t]
        """
        # Initialize empty matrix
        uniform_samples = np.zeros((n, t))

        for nn in range(n):
            # Random initialization (get the chain going)
            u = stats.uniform.rvs(0, 1, random_state=rng)
            for tt in range(t):
                # Do not provide w to get a random sample
                v = self.variable(u=u, w=None, rng=rng)
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
        p *= (u ** -self.alpha + v ** -self.alpha - 1) ** (-1 / self.alpha - 2)

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

        # NOTE: in the following term, the book has a sign wrong, this has
        # been corrected here:
        exp_num = - self.alpha**2 * \
            (psi_i_u**2 + psi_i_v**2) + 2 * self.alpha * psi_i_u * psi_i_v
        exp_den = 2 * (1 - self.alpha**2)

        # Bivariate density
        return term1 * np.exp(exp_num / exp_den)
