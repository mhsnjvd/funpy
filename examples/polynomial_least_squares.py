import numpy as np
import matplotlib.pyplot as plt
from polyfit import polyfit

n_pts = 101
x = np.linspace(-1, 1, n_pts)
y = 1.0 / (1.0 + 25.0 * x**2) + 1.0e-1*np.random.randn(n_pts)
n = 15
f = polyfit(y, x, n, [-1, 1])
plt.plot(x, y, 'xr')
f.plot()
plt.title(f'Discrete polynomial least-squares fit of degree {n}')
