from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are
from spatialmath import SO2
from spatialmath.base import angle_wrap

from koopman.simulation.animation import PlotElement, PlotEnvironment


class DynamicalSystem:
    nx = None
    nu = None
    nz = None

    def __init__(self, name: Optional[str] = None, params: Optional[Any] = None) -> None:
        self.name = name
        self.params = params

        self.t_history = self.x_history = self.u_history = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Check if the subclass has its own definition of 'a'
        if cls.nx is DynamicalSystem.nx or cls.nu is DynamicalSystem.nu:
            raise NotImplementedError(f"Class variable 'nx' and 'nu' must be overridden in {cls.__name__}")

    def project_state(self, x: np.ndarray) -> np.ndarray:
        return x

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    #     return self.batch_dynamics(x[None, :], u[None, :])[0, ...]

    def koopman_observables(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def clear_history(self) -> None:
        self.t_history = self.x_history = self.u_history = None

    def set_history(self, ts, xs, us) -> None:
        self.t_history = ts
        self.x_history = xs
        self.u_history = us

    def get_history(self) -> Any:
        return (
            np.asarray(self.t_history),
            np.asarray(self.x_history),
            np.asarray(self.u_history),
        )

    def interp_states(self, t, x0, x1):
        return t * x0 + (1 - t) * x1

    def query_history(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        assert self.t_history is not None, "No time history to query"

        # Clamp t to the range of the time history
        t = max(0, min(self.t_history[-1], t))
        idx_hi = np.searchsorted(self.t_history, t)
        idx_lo = max(0, idx_hi - 1)

        # Look up indices based on time to do linear interpolation of states, controls
        t_lo, t_hi = self.t_history[idx_lo], self.t_history[idx_hi]

        if idx_lo == idx_hi:
            alpha_lo, alpha_hi = 0.0, 1.0
        else:
            alpha_hi = (t - t_lo) / (t_hi - t_lo)
            alpha_lo = 1.0 - alpha_hi

        x_interp = self.interp_states(alpha_lo, self.x_history[idx_lo], self.x_history[idx_hi])

        if idx_hi >= len(self.u_history):
            idx_lo = idx_hi = len(self.u_history) - 1
            alpha_hi, alpha_lo = 1.0, 0.0

        # Zero-order hold for control inputs
        u_interp = self.u_history[idx_lo]

        return np.asarray(x_interp), np.asarray(u_interp)


class NonlinearAttractor2D(DynamicalSystem):
    @dataclass
    class Params:
        mu: float
        lam: float

    nx = 2
    nu = 1

    def __init__(self, params: Params) -> None:
        super().__init__("NonlinearAttractor2D", params)

    def dynamics(self, x, u):
        is_batch = len(x.shape) == 2

        if not is_batch:
            x = np.expand_dims(x, axis=0)

        x1 = x[:, 0, None]
        x2 = x[:, 1, None]

        x1_dot = -self.params.mu * x1
        x2_dot = -self.params.lam * (x2 - x1 ** 2) + u

        out = np.column_stack([x1_dot, x2_dot])

        if is_batch:
            return out
        else:
            return np.squeeze(out, axis=0)


class VanDerPolOscillator(DynamicalSystem):
    nx = 2
    nu = 1

    def __init__(self) -> None:
        super().__init__("VanDerPolOscillator", None)

    def dynamics(self, x, u):
        is_batch = len(x.shape) == 2

        if not is_batch:
            x = np.expand_dims(x, axis=0)

        x1 = x[:, 0, None]
        x2 = x[:, 1, None]

        x1_dot = 2.0 * x2
        x2_dot = -0.8 * x1 + 2.0 * x2 - 10.0 * x1 ** 2 * x2 + u

        out = np.column_stack([x1_dot, x2_dot])

        if is_batch:
            return out
        else:
            return np.squeeze(out, axis=0)


class Pendulum(DynamicalSystem):
    @dataclass
    class Params:
        m: float  # Mass of the pendulum
        l: float  # Length of the pendulum
        g: float  # Gravity
        b: float  # Damping coefficient

    nx = 2  # [theta, theta_dot]
    nu = 1  # Torque input

    def __init__(self, params: Params) -> None:
        super().__init__("Pendulum", params)

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        is_batch = len(x.shape) == 2

        if not is_batch:
            x = np.expand_dims(x, axis=0)

        N1, nx = x.shape
        N2, nu = u.shape

        assert nx == self.nx
        assert nu == self.nu
        assert N1 == N2  # Batch size must match

        theta, theta_dot = x[:, 0, None], x[:, 1, None]  # Extract state variables

        # Compute angular acceleration
        theta_ddot = (-self.params.g / self.params.l * np.sin(theta) - self.params.b /
                      (self.params.m * self.params.l ** 2) * theta_dot + u / (self.params.m * self.params.l ** 2))

        xdot = np.column_stack([theta_dot, theta_ddot])

        if is_batch:
            return xdot
        else:
            return np.squeeze(xdot, axis=0)  # [theta_dot, theta_ddot]

    class PlotElement(PlotElement):
        def __init__(self, env: PlotEnvironment, sys: "Pendulum", rod_color='black') -> None:
            super().__init__(env)

            self.sys = sys
            self.l = sys.params.l  # Length of the pendulum

            # Draw the pendulum rod and bob
            (self.rod, ) = self.env.ax.plot([], [], 'o-', lw=2, markersize=5, c=rod_color, markerfacecolor='gray')

            # Set axis limits based on the pendulum length
            self.env.set_xlim(-self.l * 1.2, self.l * 1.2)
            self.env.set_ylim(-self.l * 1.2, self.l * 1.2)

        def update(self, t):
            state, _ = self.sys.query_history(t)  # Get the current state of the pendulum
            theta = state[0]  # Extract the pendulum angle

            # Compute pendulum end position
            pole_x = self.l * np.sin(theta)
            pole_y = -self.l * np.cos(theta)

            # Update pendulum rod and bob
            self.rod.set_data([0, pole_x], [0, pole_y])


class CartPole(DynamicalSystem):
    @dataclass
    class Params:
        m_c: float
        m_p: float
        l: float
        g: float

    nx = 4
    nu = 1

    def __init__(self, params: Params) -> None:
        super().__init__("CartPole", params)

    # def project_state(self, x: np.ndarray) -> np.ndarray:
    #     # Project the state to a suitable range if necessary
    #     # For CartPole, we might want to wrap the angle theta to be within [-pi, pi]
    #     if len(x.shape) == 2:
    #         theta = x[:, 1]
    #         x[:, 1] = np.remainder(theta, 2 * np.pi)
    #     else:
    #         theta = x[1]
    #         x[1] = theta % (2 * np.pi)

    #     return x

    def batch_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        N1, nx = x.shape
        N2, nu = u.shape

        assert nx == self.nx
        assert nu == self.nu
        assert N1 == N2

        theta, v, theta_dot = x[:, 1, None], x[:, 2, None], x[:, 3, None]

        # Compute accelerations
        x_ddot = (1 / (self.params.m_c + self.params.m_p * np.sin(theta) ** 2)) * (
            u + self.params.m_p * np.sin(theta) * (self.params.l * theta_dot ** 2 + self.params.g * np.cos(theta)))

        theta_ddot = (1 / (self.params.l * (self.params.m_c + self.params.m_p * np.sin(theta) ** 2))) * (
            -u * np.cos(theta) - self.params.m_p * self.params.l * theta_dot ** 2 * np.cos(theta) * np.sin(theta) -
            (self.params.m_c + self.params.m_p) * self.params.g * np.sin(theta))

        # Stack results into an output array
        return np.column_stack([v, theta_dot, x_ddot, theta_ddot])

    class PlotElement(PlotElement):
        def __init__(self, env: PlotEnvironment, sys: "CartPole", cart_color='blue') -> None:
            super().__init__(env)

            self.sys = sys

            self.cart_width, self.cart_height = 0.4, 0.2

            self.cart = self.env.ax.add_patch(
                plt.Rectangle((-self.cart_width / 2, -self.cart_height / 2),
                              self.cart_width,
                              self.cart_height,
                              fc=cart_color))  # Cart
            (self.rod, ) = self.env.ax.plot([], [], 'o-', lw=2, markersize=5, c='black', markerfacecolor='gray')  # Pole

            x_lo, x_hi = sys.x_history[:, 0].min(), sys.x_history[:, 0].max()
            new_range = (x_hi - x_lo + self.cart_width) * 1.2

            x_lo = (x_lo + x_hi) / 2 - new_range / 2
            x_hi = (x_lo + x_hi) / 2 + new_range / 2

            self.env.set_xlim(x_lo, x_hi)
            self.env.set_ylim(-sys.params.l * 1.2, sys.params.l * 1.2)

        def update(self, t):
            state, _ = self.sys.query_history(t)  # Get the current state of the cartpole
            x, theta = state[0], state[1]  # Extract cart position and pole angle

            # Compute the pole's end position
            l = self.sys.params.l
            pole_x = x + l * np.sin(theta)
            pole_y = -l * np.cos(theta)

            # Update cart position
            self.cart.set_xy((x - 0.2, -0.1))  # Adjusted for cart size

            # Update pole position
            self.rod.set_data([x, pole_x], [0, pole_y])


if __name__ == "__main__":
    cart_pole = CartPole(params=CartPole.Params(1, 1, 1, 9.81))

    xbar = np.array([0, np.pi - 0.4, 0, 0], dtype=np.float32)
    ubar = np.array([24.24184], dtype=np.float32)

    print(cart_pole.dynamics(xbar, ubar))

    xbar = np.tile(xbar, (5, 1))
    ubar = np.tile(ubar, (5, 1))

    print(cart_pole.batch_dynamics(xbar, ubar))
