"""
Initial shift examples.

"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from triage.generate_timeseries import LinearSSM


def avg_acf(ts):
    """
    Helper function to visualize average acf
    """
    return np.mean([sm.tsa.acf(x) for x in ts], axis=0)


def plot_ts_and_new_ts(ts, new_ts):
    """
    Factor out the comparison plots
    """
    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    # Create the subplots
    ax1 = fig.add_subplot(grid[0, 0])  # Top left (spans 2 columns)
    ax2 = fig.add_subplot(grid[0, 1])   # Top right
    ax3 = fig.add_subplot(grid[1, 0])  # Bottom left (spans 2 columns)
    ax4 = fig.add_subplot(grid[1, 1])   # Bottom right

    ax1.plot(ts[:, 1, :].T)
    ax1.set_title('Training data')

    ax2.plot(avg_acf(ts[:, 1, :]))
    ax2.set_title('Training autocorr.')

    ax3.plot(new_ts[:, 1, :].T)
    ax3.set_title('Shifted data')

    ax4.plot(avg_acf(new_ts[:, 1, :]))
    ax4.set_title('Shifted autocorr.')

    plt.tight_layout()
    plt.show()


def run_example():
    """
    Driver function to run this example

    """
    # Settings
    rng = np.random.default_rng(seed=42)
    d = 10
    p = 2
    n = 10
    t = 150

    # Generate "training data"
    state_matrix = rng.normal(size=(d, d))
    obs_matrix = rng.normal(size=(p, d))
    state_noise_cov = 0.1 * np.eye(d)
    obs_noise_cov = 0.01 * np.eye(p)

    # Instantiate the SSM object
    ssm = LinearSSM(state_matrix, state_noise_cov, obs_matrix, obs_noise_cov,
                    rng=rng, scale_matrix=True)

    # Sample the time series
    ssm.sample(n, t)
    ssm.plot_sample()

    # Case A: Generate with changed state matrix
    state_matrix_A = ssm.state_matrix + 0.1 * rng.normal(size=(d, d))
    ssm_A = ssm.copy_with_changes(state_matrix=state_matrix_A,
                                  scale_matrix=True)

    ssm_A.sample(n, t)
    plot_ts_and_new_ts(ssm.ts_samples, new_ts=ssm_A.ts_samples)

    # Case B: change observation noise
    obs_noise_cov_B = 1.0 * np.eye(p)
    ssm_B = ssm.copy_with_changes(obs_noise_cov=obs_noise_cov_B)

    ssm_B.sample(n, t)
    plot_ts_and_new_ts(ssm.ts_samples, new_ts=ssm_B.ts_samples)

    # Case C: change observation matrix
    obs_matrix_C = ssm.obs_matrix + 1.5 * rng.normal(size=(p, d))
    ssm_C = ssm.copy_with_changes(obs_matrix=obs_matrix_C)

    ssm_C.sample(n, t)
    plot_ts_and_new_ts(ssm.ts_samples, new_ts=ssm_C.ts_samples)

    # Case D: change point (sort of)
    # TODO
