import copy
from dataclasses import dataclass

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray


@dataclass
class PermutationDistance(object):
    """
    Class that aggregates all detection methods
    Parameters:
    - train_data: A list of training data samples.
    - target_data: A list of target data samples.
    - p_value: P value; default is 0.05.
    - num_samples: Number of points to sample for estimating distribution;
        default is 500.
    - num_bins: Number of bins for binning the data for calculating PSI;
        default is 20.
    - min_bin_prob: Minimum probability values used for a bin to avoid
        issues with empty bin (divide by 0 issues); default is 1*10^-10.
    - n_resamples: Number of permutations evaluated for the permutation
        test; default is 1,000.
    - diff_tol: Difference tolerance to consider data different for
        memoization purposes
    """

    train_data: NDArray | list[float]
    target_data: NDArray | list[float]

    num_bins: int = 20
    num_samples: int = 500
    p_value: float = 5e-2
    min_bin_prob: float = 1e-10
    n_resamples: int = 1000
    wass_p: int = 1
    diff_tol: float = 1e-14

    def __post_init__(self) -> None:
        """
        These are variables used for resampling memoization. That way both
        "self.train_data" and "self.target_data" can be kept unchanged from
        one reshufling to the next. They also help to recognize when we cannot
        rely on memoization and have to re-caculate the expensive values again.
        """

        self.train_d: NDArray | None = None
        self.target_d: NDArray | None = None
        self.prev_train: NDArray | None = None
        self.prev_target: NDArray | None = None

    def __data_range(self, train_data: ArrayLike | None = None,
                     target_data: ArrayLike | None = None
                     ) -> tuple[float, float]:
        """
        Retreive the data range

        Parameters:
        - train_data: A list of training data samples. Default value is given
            at initialization.
        - target_data: A list of target data samples. Default value is given
            at initialization.
        """

        if train_data is None:
            train_data = self.train_data
        if target_data is None:
            target_data = self.target_data

        # Get maxima and minima of data
        all_data = np.concatenate((train_data, target_data))
        return np.min(all_data), np.max(all_data)

    def __get_percent(self, train_data: ArrayLike | None = None,
                      target_data: ArrayLike | None = None,
                      num_bins: int | None = None,
                      min_bin_prob: float | None = None
                      ) -> tuple[NDArray, NDArray]:
        """
        Common operations for histogram calculation

        Parameters:
        - train_data: A list of training data samples. Default value is given
            at initialization.
        - target_data: A list of target data samples. Default value is given
            at initialization.
        - num_bins: Number of bins for binning the data for calculating PSI;
            default is 20.
        - min_bin_prob: Minimum probability values used for a bin to avoid
            issues with empty bin (divide by 0 issues); default is 1e-10.
        """

        if train_data is None:
            train_data = self.train_data
        if target_data is None:
            target_data = self.target_data
        if num_bins is None:
            num_bins = self.num_bins
        if min_bin_prob is None:
            min_bin_prob = self.min_bin_prob

        min_, max_ = self.__data_range(train_data=train_data,
                                       target_data=target_data)

        # Train data percent
        train_p = np.histogram(train_data, bins=num_bins,
                               range=(min_, max_))[0]
        train_p /= np.size(train_data)
        train_p = np.clip(train_p, a_min=min_bin_prob, a_max=None)

        # Target data percent
        target_p = np.histogram(target_data, bins=num_bins,
                                range=(min_, max_))[0]
        target_p /= np.size(target_data)
        target_p = np.clip(target_p, a_min=min_bin_prob, a_max=None)

        return train_p, target_p

    def __get_distrib(self, train_data: ArrayLike | None = None,
                      target_data: ArrayLike | None = None,
                      num_samples: int | None = None
                      ) -> tuple[NDArray, NDArray]:
        """
        Common operations for kde calculation

        Parameters:
        - train_data: A list of training data samples. Default value is given
            at initialization.
        - target_data: A list of target data samples. Default value is given
            at initialization.
        - num_samples: Number of points to sample for estimating distribution;
            default=1000.
        """

        if train_data is None:
            train_data = self.train_data
        if target_data is None:
            target_data = self.target_data
        if num_samples is None:
            num_samples = self.num_samples

        min_, max_ = self.__data_range(train_data=train_data,
                                       target_data=target_data)

        vals = np.linspace(min_, max_, num_samples)

        # Train data distribution
        train_d = sp.stats.gaussian_kde(train_data).evaluate(vals)

        # Target data distribution
        target_d = sp.stats.gaussian_kde(target_data).evaluate(vals)

        return train_d, target_d

    def metric(self, type_: str, use: str,
               train_data: ArrayLike | None = None,
               target_data: ArrayLike | None = None,
               num_bins: int | None = None,
               min_bin_prob: float | None = None,
               num_samples: int | None = None,
               wass_p: float | None = None,
               memoize: bool = False) -> float:
        """
        Returns a given metric

        Parameters:
        - type_: String, metric type, values: "PSI", "JS", "WD", "KS"
        - use: String, either "histogram" or "kde"
        - train_data: A list of training data samples. Default value is given
            at initialization.
        - target_data: A list of target data samples. Default value is given
            at initialization.
        - num_samples: Number of points to sample for estimating distribution;
            default value is given at initialization.
        - num_bins: Number of bins for binning the data for calculating PSI;
            default value is given at initialization.
        - min_bin_prob: Minimum probability values used for a bin to avoid
            issues with empty bin (divide by 0 issues); default value is given
            at initialization.
        - wass_p: Wasserstein distance order; default default value is given
            at initialization.
        - memoize: Wether to memoize or not, defaults to False
        """

        assert type_ in ["PSI", "JS", "WD", "KS"]
        assert use in ["histogram", "kde"]

        if train_data is None:
            train_data = self.train_data
        if target_data is None:
            target_data = self.target_data

        need_dist_estimates = type_ in ["PSI", "JS"]

        if need_dist_estimates:
            # need to calculate distribution estimates
            if not memoize:
                self.train_d = None
                self.target_d = None
            else:
                # Here we are checking if the values have changed, even if
                # asked to memoize, the values should be calculated again if
                # anything has changed.
                #
                # If we do not have previous data, always calculate
                if self.prev_train is None:
                    self.train_d = None
                elif (abs(self.prev_train - train_data) > self.diff_tol).any():
                    self.train_d = None

                # Same as before but with the target data
                if self.prev_target is None:
                    self.target_d = None
                elif (abs(self.prev_target - target_data) >
                      self.diff_tol).any():
                    self.target_d = None

                # Remember the new data
                assert type(target_data) is np.ndarray
                assert type(train_data) is np.ndarray
                self.prev_target = copy.copy(target_data)
                self.prev_train = copy.copy(train_data)

            # Train and target percentages or distributions
            if self.train_d is None or self.target_d is None:
                if use == "histogram":
                    tup = self.__get_percent(train_data=train_data,
                                             target_data=target_data,
                                             num_bins=num_bins,
                                             min_bin_prob=min_bin_prob)
                elif use == "kde":
                    tup = self.__get_distrib(train_data=train_data,
                                             target_data=target_data,
                                             num_samples=num_samples)
                else:
                    e = "Provide a use parameter: 'histogram' or 'kde'"
                    raise Exception(e)

                self.train_d, self.target_d = tup

        assert self.train_d is not None and self.target_d is not None

        match type_:
            case "PSI":
                return np.sum((self.train_d - self.target_d) *
                              np.log(self.train_d / self.target_d))
            case "JS":
                return sp.spatial.distance.jensenshannon(self.train_d,
                                                         self.target_d,
                                                         base=np.e)
            case "WD":
                if wass_p is None:
                    wass_p = self.wass_p
                return np.mean(
                    np.abs(
                        np.sort(train_data) - np.sort(target_data)
                    ) ** wass_p) ** (-wass_p)
            case "KS":
                return sp.stats.kstest(train_data,  # type: ignore[arg-type]
                                       target_data,  # type: ignore[arg-type]
                                       alternative='two-sided').statistic
            case _:
                e = "Provide a type_ parameter: 'PSI', 'JS', 'WD', or 'KS'"
                raise Exception(e)

    def data_shift_test(self, use: str, metrics: list[str] | str,
                        wass_p: float | None = None,
                        num_bins: int | None = None,
                        min_bin_prob: float | None = None,
                        num_samples: int | None = None,
                        n_resamples: int | None = None,
                        p_value: float | None = None
                        ) -> list[tuple[float, bool]] | tuple[float, bool]:
        """
        Return tuple of p_value and boolean value equal to 1 if data shift is
        detected from "self.train_data" to "self.target_data" based on metric
        and permutation test, else 0.

        If list of metrics was given, return list of the tuples described above
        for each type of metric in the same order.

        Parameters:
        - use: String, either "histogram" or "kde"
        - metrics: String or list of strings, metric type, values: "PSI", "JS",
            "WD", "KS"
        - wass_p: Wasserstein distance order; default is 1
        - num_bins: Number of bins for binning the data for calculating PSI;
            default is 20.
        - min_bin_prob: Minimum probability values used for a bin to avoid
            issues with empty bin (divide by 0 issues); default is 1*10^-10.
        - num_samples: Number of points to sample for estimating distribution;
            default is 1000.
        - n_resamples: Number of permutations evaluated for the permutation
            test; default is 100,000.
        - p_value: P value; default is 0.05.
        """

        assert use in ["histogram", "kde"]

        if n_resamples is None:
            n_resamples = self.n_resamples
        if p_value is None:
            p_value = self.p_value
        if wass_p is None:
            wass_p = self.wass_p

        is_list = type(metrics) is list

        if is_list:
            metrics_list = metrics
        else:
            assert type(metrics) is str
            metrics_list = [metrics]

        observed_stat: list[float] = []
        permuted_stats: list[list[float]] = []
        results: list[tuple[float, bool]] = []

        # Save the combined data here
        combined_data = np.concatenate((self.train_data, self.target_data))

        for type_ in metrics_list:

            observed_stat.append(0.)
            permuted_stats.append([])
            results.append((0., False))

            if type_ == "KS":
                ks_test_result = sp.stats.kstest(self.train_data,
                                                 self.target_data,
                                                 alternative='two-sided')

                # Give the results already
                results[-1] = (ks_test_result.pvalue,
                               ks_test_result.pvalue < p_value)
                continue

            observed_stat[-1] = self.metric(type_, use, wass_p=wass_p,
                                            num_bins=num_bins,
                                            min_bin_prob=min_bin_prob,
                                            num_samples=num_samples)

        for i in range(n_resamples):
            np.random.shuffle(combined_data)
            perm_group_a = combined_data[:len(self.train_data)]
            perm_group_b = combined_data[len(self.train_data):]

            for j, type_ in enumerate(metrics_list):
                # Ignore KS, we have that already
                if type_ == "KS":
                    continue

                permuted_stats[j].append(self.metric(type_, use,
                                                     train_data=perm_group_a,
                                                     target_data=perm_group_b,
                                                     wass_p=wass_p,
                                                     num_bins=num_bins,
                                                     min_bin_prob=min_bin_prob,
                                                     num_samples=num_samples,
                                                     memoize=True))

        for j, type_ in enumerate(metrics_list):
            if type_ == "KS":
                continue

            test_p_value = np.mean(
                np.abs(permuted_stats[j]) >= np.abs(observed_stat[j]))
            result = (test_p_value, test_p_value < p_value)
            results[j] = result

        if is_list:
            return results
        else:
            return results[0]
