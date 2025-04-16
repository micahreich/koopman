import numpy as np
import matplotlib.pyplot as plt
from koopman.simulation.simulator import simulate, simulate_batch
from koopman.simulation.systems import DynamicalSystem, Pendulum
from koopman.edmd.edmd import eDMD
from scipy.ndimage import gaussian_filter1d

def generate_random_smooth_controls(N, T):
    dt = 0.05
    w = np.random.randn(N, T, 1) * np.sqrt(dt)
    b = np.cumsum(w, axis=1)
    b_smooth = gaussian_filter1d(b, sigma=10, axis=1)

    return b_smooth

def compute_rbf_observables(X, centers, sigma):
    # X: (N_samples, state_dim)
    # centers: (N_rbf_per_state_var)
    diffs = X[:, :, None] - centers[None, None, :]  # (N_samples, state_dim, N_rbf_per_state_var)
    sq_dists = diffs ** 2                           # "
    return np.exp(-sq_dists / (2 * sigma**2)).reshape(X.shape[0], -1) # (N_samples, N_rbf)

if __name__ == '__main__':

    pendulum = Pendulum(Pendulum.Params(m=1, l=1, g=9.81, b=0.0))

    tf = 5.0
    dt = 0.05
    N = 5_000

    # IC
    theta0 = np.random.uniform(-np.pi, np.pi, (N, 1))
    omega0 = np.random.uniform(-3, 3, (N, 1))
    x0 = np.hstack((theta0, omega0))

    U = generate_random_smooth_controls(N, int(tf/dt))

    ts, xhist, uhist = simulate_batch(
        sys=pendulum,
        tf=tf,
        dt=dt,
        u=U,
        x0=x0
    )

    print("ts:",    ts.shape)     # Nt,
    print("xhist:", xhist.shape)  # N, Nt, dim(x_k)
    print("uhist:", uhist.shape)  # N, Nt, dim(x_k)

    edmd = eDMD(1, 2, bilinear=True)

    # RBF observables
    N_rbf_per_state_var = 20
    rbf_min_center = min(np.min(U), -np.pi)
    rbf_max_center = max(np.max(U),  np.pi)
    sigma = 0.5 * (rbf_max_center - rbf_min_center) / N_rbf_per_state_var
    centers = np.linspace(rbf_min_center, rbf_max_center, N_rbf_per_state_var)

    # observables function: (M, nx) -> (M, N_obs)
    koopman_observables = lambda X: np.column_stack([X, compute_rbf_observables(X, centers, sigma)])

    zhist = edmd.apply_observable_to_history(xhist, koopman_observables)

    print(f"zhist.shape: {zhist.shape}")

    Kpa, Kpb, Kpc = edmd.fit(zhist, uhist)
    print(f"Kpa: {Kpa}")
    print(f"Kpb: {Kpb}")
    print(f"Kpc: {Kpc}")

    # Evaluate

    theta0 = np.random.uniform(-np.pi, np.pi, 1)
    omega0 = np.random.uniform(-1, 1.0, 1)
    x0 = np.concatenate((theta0, omega0))
    controls = np.zeros((int(tf/dt), 1)) #np.squeeze(generate_random_smooth_controls(1, int(tf/dt)), axis=0)

    eval_ts, eval_xhist, eval_uhist = simulate(
        sys=pendulum,
        tf=tf,
        dt=dt,
        u=controls,
        x0=x0
    )

    zjm1 = koopman_observables(np.atleast_2d(x0)).reshape(-1)

    eval_xhist_pred = np.empty_like(eval_xhist)
    eval_xhist_pred[0] = edmd.project_to_x(zjm1)

    for i, t in enumerate(eval_ts[:-1]):
        zj = edmd.predict_z_next(zjm1, controls[i])
        eval_xhist_pred[i + 1] = edmd.project_to_x(zj)

        zjm1 = zj


    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(eval_ts, eval_xhist[:, 0], color='red', label="True theta")
    ax[0].plot(eval_ts, eval_xhist_pred[:, 0], color='red', label="Predicted theta", linestyle='--')
    ax[0].plot(eval_ts, eval_xhist[:, 1], color='blue', label="True omega")
    ax[0].plot(eval_ts, eval_xhist_pred[:, 1], color='blue', label="Predicted omega", linestyle='--')
    ax[0].set_title("Pendulum Angle")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Theta (rad)")
    ax[0].legend()
    plt.tight_layout()
    plt.show()

