import matplotlib.pyplot as plt
import numpy as np

from triage import HSMM


def run_example():
    """
    Example for the HSMM class
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
    hsmm = HSMM(init_probs, transition_probs, emission_means,
                emission_covariances, state_durations_params)

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
            axes[ax_index].plot(data, label=f"{label} {i + 1}", color=color)
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
