# Physical pendulum with quadratic cost and non-convex constraints



if __name__ == "__main__":

    params = Pendulum.Params(
        m=1, l=1, g=9.81, b=0.0
    )

    pendulum = Pendulum(params)

    tf = 5.0
    dt = 0.05
    N = 5_000

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

    split = 0.8
    N_train = int(N * split)
    N_eval = N - N_train

    xhist_train, uhist_train = xhist[:N_train], uhist[:N_train]
    xhist_eval, uhist_eval = uhist[N_train:], uhist[N_train:]

    
