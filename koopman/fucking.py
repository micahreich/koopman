import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

dt = 0.1
tf = 6
n = 250
ts = np.linspace(0, tf, n)

w = np.random.randn(n) * np.sqrt(dt)  # white noise
b = np.cumsum(w)  # Brownian motion
w_smooth = gaussian_filter1d(b, sigma=10)
b_smooth = w_smooth * np.sin(0.5 * 2 * np.pi * ts)

plt.plot(ts, b_smooth)
plt.title("Smoothed Brownian Motion")
plt.show()
