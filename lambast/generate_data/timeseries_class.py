from numpy.typing import NDArray


class TimeSeries(object):

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
