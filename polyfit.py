import numpy as np
from function import Function
from numba import njit


@njit
def polyfit_jit(y: np.array, x: np.array, n: int, a: float, b: float) -> tuple:
    """Jitted version of polyfit, see polyfit for details
    :param y: data
    :param x: nodes
    :param n: degree of the polynomial fit
    :param a: left end point of domain
    :param b: right end point of domain
    :return: the output of np.linalg.lstsq()
    """
    m = len(x)
    Tx = np.zeros((m, n + 1))
    Tx[:, 0] = np.ones(m)
    x_map = 2.0 * (x - a) / (b - a) - 1.0
    Tx[:, 1] = x_map
    for k in range(1, n):
        Tx[:, k + 1] = 2.0 * x_map * Tx[:, k] - Tx[:, k-1]

    # TODO: this is not compiling :(
    # Initialize variables for jit:
    c = np.zeros((n+1,))
    residuals = np.zeros((1,))
    rank = int(0)
    singular_values = np.zeros((n+1,))

    c, residuals, rank, singular_values = np.linalg.lstsq(Tx, y, rcond=None)

    return c, residuals, rank, singular_values


def polyfit(y, x, n, domain):
    """Degree n least squares polynomial approximation of data y taken on points x
    in a domain.
    :param y: y-values, i.e., data values, np array
    :param x: x-values, np array
    :param n: degree of approximation, an integer
    :param domain: domain of approximation
    :return: 
    """

    domain = 1.0 * np.array(domain)
    a = domain[0]
    b = domain[-1]
    assert len(x) == len(y), f"len(x) = {len(x)}, while len(y) = {len(y)}, these must be equal"

    # map points to [-1, 1]
    m = len(x)
    x_normalized = 2.0 * (x - a) / (b - a) - 1.0
    # construct the Chebyshev-Vandermonde matrix:
    Tx = np.zeros((m, n+1))
    Tx[:, 0] = np.ones(m)
    Tx[:, 1] = x_normalized
    for k in range(1, n):
        Tx[:, k + 1] = 2.0 * x_normalized * Tx[:, k] - Tx[:, k-1]

    # c, residuals, rank, singular_values = polyfit_jit(y, x, n, a, b)
    c, residuals, rank, singular_values = np.linalg.lstsq(Tx, y, rcond=None)

    # Make a function:
    f = Function(coefficients=c, domain=domain)
    return f


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 11)
    y = x**2
    n = 2
    domain = [0, 10]
    a = domain[0]
    b = domain[-1]
    f = polyfit(y, x, n, domain)
    plt.plot(x, y, '.')
    f.plot()
