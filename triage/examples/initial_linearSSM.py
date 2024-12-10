"""
Initial shift examples.

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from triage.generate_timeseries import LinearSSM


def avg_acf(ts):
    """
    Helper function to visualize average acf
    """
    acf = sm.tsa.acf(ts[0, :])
    for i in range(1, len(ts)):
        acf += sm.tsa.acf(ts[i, :])

    return acf/len(ts)


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

    ssm = LinearSSM(state_matrix, state_noise_cov, obs_matrix, obs_noise_cov,
                    rng=rng)
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

    # Case A: Generate with changed state matrix
    state_matrix_A = state_matrix + 0.1*rng.normal(size=(d, d))
    eigenvalues = np.linalg.eigvals(state_matrix_A)
    spectral_radius = max(abs(eigenvalues))

    # Scale to ensure spectral radius < 1
    if spectral_radius >= 1:
        state_matrix_A = state_matrix_A / (spectral_radius + 0.1)

    obs_matrix_A = obs_matrix
    state_noise_cov_A = 0.1 * np.eye(d)
    obs_noise_cov_A = 0.01 * np.eye(p)

    ssm_A = LinearSSM(state_matrix_A, state_noise_cov_A, obs_matrix_A,
                      obs_noise_cov_A, rng=rng)
    ts_A = ssm_A.sample(n, t)

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
    ax3.plot(ts_A[:, 1, :].T)
    ax3.set_title('Shifted data')
    ax4.plot(avg_acf(ts_A[:, 1, :]))
    ax4.set_title('Shifted autocorr.')
    plt.tight_layout()
    plt.show()

    # Case B: change observation noise
    state_matrix_B = state_matrix
    obs_matrix_B = obs_matrix
    state_noise_cov_B = 0.1 * np.eye(d)
    obs_noise_cov_B = 1.0 * np.eye(p)

    ssm_B = LinearSSM(state_matrix_B, state_noise_cov_B, obs_matrix_B,
                      obs_noise_cov_B, rng=rng)
    ts_B = ssm_B.sample(n, t)

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
    ax3.plot(ts_B[:, 1, :].T)
    ax3.set_title('Shifted data')
    ax4.plot(avg_acf(ts_B[:, 1, :]))
    ax4.set_title('Shifted autocorr.')
    plt.tight_layout()
    plt.show()

    # Case C: change observation matrix
    state_matrix_C = state_matrix
    obs_matrix_C = obs_matrix + 1.5*rng.normal(size=(p, d))
    state_noise_cov_C = 0.1 * np.eye(d)
    obs_noise_cov_C = 0.01 * np.eye(p)

    ssm_C = LinearSSM(state_matrix_C, state_noise_cov_C, obs_matrix_C,
                      obs_noise_cov_C, rng=rng)
    ts_C = ssm_C.sample(n, t)

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
    ax3.plot(ts_C[:, 1, :].T)
    ax3.set_title('Shifted data')
    ax4.plot(avg_acf(ts_C[:, 1, :]))
    ax4.set_title('Shifted autocorr.')
    plt.tight_layout()
    plt.show()

    # Case D: change point (sort of)
    # TODO
