import copy

import numpy as np
import scipy as sp


class DetectionMethods(object):
    """
    Class that aggregates all detection methods
    """

    def __init__(self, train_data, target_data, num_bins=20, num_samples=1000,
                 p_value=5e-2, min_bin_prob=1e-10, n_resamples=1e5, wass_p=1,
                 diff_tol=1e-14):
        """
        Parameters:
        - train_data: A list of training data samples.
        - target_data: A list of target data samples.
        - p_value: P value; default is 0.05.
        - num_samples: Number of points to sample for estimating distribution;
            default=1000.
        - num_bins: Number of bins for binning the data for calculating PSI;
            default is 20.
        - min_bin_prob: Minimum probability values used for a bin to avoid
            issues with empty bin (divide by 0 issues); default is 1*10^-10.
        - n_resamples: Number of permutations evaluated for the permutation
            test; default is 100,000.
        - diff_tol: Difference tolerance to consider data different for
            memoization purposes
        """

        self.train_data = train_data
        self.target_data = target_data

        self.num_bins = num_bins
        self.num_samples = num_samples
        self.p_value = p_value
        self.min_bin_prob = min_bin_prob
        self.n_resamples = int(n_resamples)
        self.wass_p = wass_p

        # NOTE: These are variables used for resampling memoization. That way
        # both "self.train_data" and "self.target_data" can be kept unchanged
        # from one reshufling to the next. They also help to recognize when
        # we cannot rely on memoization and have to re-caculate the expensive
        # values again.
        self.train_d = None
        self.target_d = None
        self.prev_train = None
        self.prev_target = None
        self.diff_tol = diff_tol

    def __data_range(self, train_data=None, target_data=None):
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
        return np.max(all_data), np.min(all_data)

    def __get_percent(self, train_data=None, target_data=None, num_bins=None,
                      min_bin_prob=None):
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
            issues with empty bin (divide by 0 issues); default is 1*10^-10.
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

    def __get_distrib(self, train_data=None, target_data=None,
                      num_samples=None):
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

        _, max_ = self.__data_range(train_data=train_data,
                                    target_data=target_data)

        vals = np.linspace(0, max_, num_samples)

        # Train data distribution
        train_d = sp.stats.gaussian_kde(train_data).evaluate(vals)

        # Target data distribution
        target_d = sp.stats.gaussian_kde(target_data).evaluate(vals)

        return train_d, target_d

    def metric(self, train_data=None, target_data=None, num_bins=None,
               min_bin_prob=None, num_samples=None, wass_p=None, use=None,
               type_=None, memoize=False):
        """
        Returns a given metric

        Parameters:
        - train_data: A list of training data samples. Default value is given
            at initialization.
        - target_data: A list of target data samples. Default value is given
            at initialization.
        - num_samples: Number of points to sample for estimating distribution;
            default=1000.
        - num_bins: Number of bins for binning the data for calculating PSI;
            default is 20.
        - min_bin_prob: Minimum probability values used for a bin to avoid
            issues with empty bin (divide by 0 issues); default is 1*10^-10.
        - use: String, either "histogram" or "kde"
        - type_: String, metric type, values: "PSI", "JS", "WD", "KS"
        - wass_p: Wasserstein distance order; default is 1
        """

        if train_data is None:
            train_data = self.train_data
        if target_data is None:
            target_data = self.target_data

        if not memoize:
            self.train_d = None
            self.target_d = None
        else:
            # Here we are checking if the values have changed, even if asked
            # to memoize, the values should be calculated again if anything
            # has changed.
            #
            # If we do not have previous data, always calculate
            if self.prev_train is None:
                self.train_d = None
            elif (abs(self.prev_train - train_data) > self.diff_tol).any():
                self.train_d = None

            # Same as before but with the target data
            if self.prev_target is None:
                self.target_d = None
            elif (abs(self.prev_target - target_data) > self.diff_tol).any():
                self.target_d = None

            # Remember the new data
            self.prev_target = copy.copy(target_data)
            self.prev_train = copy.copy(train_data)

        # Train and target percentages or distributions
        if self.train_d is None or self.target_d is None:
            if use == "histogram":
                train_d, target_d = self.__get_percent(train_data=train_data,
                                                       target_data=target_data,
                                                       num_bins=num_bins,
                                                       min_bin_prob=min_bin_prob)
            elif use == "kde":
                train_d, target_d = self.__get_distrib(train_data=train_data,
                                                       target_data=target_data,
                                                       num_samples=num_samples)
            elif type_ != "KS":
                e = "Provide a use parameter: 'histogram' or 'kde'"
                raise Exception(e)

            self.train_d = train_d
            self.target_d = target_d

        if type_ == "PSI":
            value = np.sum((self.train_d - self.target_d) *
                           np.log(self.train_d / self.target_d))
        elif type_ == "JS":
            value = sp.spatial.distance.jensenshannon(self.train_d,
                                                      self.target_d, base=np.e)
        elif type_ == "WD":
            if wass_p is None:
                wass_p = self.wass_p

            if wass_p == 1:
                value = sp.stats.wasserstein_distance(self.train_d,
                                                      self.target_d)
            else:
                value = np.mean(np.abs(self.train_d - self.target_d) ** wass_p)
                value **= 1 / wass_p
        elif type_ == "KS":
            ks_test_result = sp.stats.kstest(train_data, target_data,
                                             alternative='two-sided')
            value = ks_test_result.statistic
        else:
            e = "Provide a type_ parameter: 'PSI', 'JS', or 'WD'"
            raise Exception(e)

        return value

    def data_shift_test(self, wass_p=None, num_bins=None, min_bin_prob=None,
                        num_samples=None, n_resamples=None, p_value=None,
                        use=None, metrics=None):
        """
        Return tuple of p_value and boolean value equal to 1 if data shift is
        detected from "self.train_data" to "self.target_data" based on metric
        and permutation test, else 0.

        If list of metrics was given, return list of the tuples described above
        for each type of metric in the same order.

        Parameters:
        - p_value: P value; default is 0.05.
        - num_samples: Number of points to sample for estimating distribution;
            default=1000.
        - num_bins: Number of bins for binning the data for calculating PSI;
            default is 20.
        - min_bin_prob: Minimum probability values used for a bin to avoid
            issues with empty bin (divide by 0 issues); default is 1*10^-10.
        - n_resamples: Number of permutations evaluated for the permutation
            test; default is 100,000.
        - use: String, either "histogram" or "kde"
        - metrics: String or list of strings, metric type, values: "PSI", "JS",
            "WD", "KS"
        - wass_p: Wasserstein distance order; default is 1
        """

        if n_resamples is None:
            n_resamples = self.n_resamples
        if p_value is None:
            p_value = self.p_value
        if wass_p is None:
            wass_p = self.wass_p

        is_list = type(metrics) is list

        if not is_list:
            metrics = [metrics]

        observed_stat = []
        permuted_stats = []
        results = []

        # Save the combined data here
        combined_data = np.concatenate((self.train_data, self.target_data))

        for type_ in metrics:

            observed_stat.append(None)
            permuted_stats.append([])
            results.append(None)

            if type_ == "KS":
                ks_test_result = sp.stats.kstest(self.train_data,
                                                 self.target_data,
                                                 alternative='two-sided')

                # Give the results already
                result = (ks_test_result.pvalue,
                          ks_test_result.pvalue < p_value)
                results[-1] = result
                continue

            observed_stat[-1] = self.metric(wass_p=wass_p, num_bins=num_bins,
                                            min_bin_prob=min_bin_prob,
                                            num_samples=num_samples, use=use,
                                            type_=type_)

        for i in range(n_resamples):
            np.random.shuffle(combined_data)
            perm_group_a = combined_data[:len(self.train_data)]
            perm_group_b = combined_data[len(self.train_data):]

            for j, type_ in enumerate(metrics):
                # Ignore KS, we have that already
                if type_ == "KS":
                    continue

                permuted_stats[j].append(self.metric(train_data=perm_group_a,
                                                     target_data=perm_group_b,
                                                     wass_p=wass_p,
                                                     num_bins=num_bins,
                                                     min_bin_prob=min_bin_prob,
                                                     num_samples=num_samples,
                                                     use=use, type_=type_,
                                                     memoize=True))

        for j, type_ in enumerate(metrics):
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
