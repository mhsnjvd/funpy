from function import Function
import numpy as np
import matplotlib.pylab as plt
from barycentric import barycentric_interpolation
from scipy.interpolate import CubicSpline

x = np.linspace(0, 5, 21)
a = 1.0
b = 4
c = 1
w = np.random.randn(len(x))
y_true = lambda x: a + b * x + c * x**2
y_noisy = lambda x: y_true(x) + w
y = y_noisy(x)


# % interpolation in equidistant points doesn't really
# work:
f = lambda xx: barycentric_interpolation(xx, y, x)
g = Function(f, length=len(y), domain=[0, 5])
g.plot()
plt.plot(x, y, 'g.', x, y - w, 'k--', x, g(x), 'r.')
plt.grid(True)
plt.show()

# % So we try cubic splines:
# x = np.arange(10)
# y = np.sin(x)
cs = CubicSpline(x, y)
xs = np.arange(0.0, 5, 0.05)
ys = y_true(xs)
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='obs data')
plt.plot(xs, ys, label='true data')
plt.plot(xs, cs(xs), label="Spline")
# plt.plot(xs, cs(xs, 1), label="S'")
# plt.plot(xs, cs(xs, 2), label="S''")
# plt.plot(xs, cs(xs, 3), label="S'''")
plt.xlim(-0.5, 5.5)
plt.legend(loc='lower left', ncol=2)
plt.show()

f = Function(lambda x: cs(x), domain=[0, 5])
g = Function(lambda x: f(x), domain=[0, 5], length=5)
f.plot()
g.plot()