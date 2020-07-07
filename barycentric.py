import numpy as np
from numba import njit


@njit
def barycentric_interpolation_jit(x: np.array,
                                  fvals: np.array,
                                  xk: np.array,
                                  wk: np.array) -> np.array:
    """See the non-jitted version for documentation. This is the
    jit enabled main workhorse code
    """
    if len(x) < 4 * len(xk):
        # Loop over evaluation points
        # Note: The value "4" here was determined experimentally.

        # Initialise return value:
        fx = np.zeros((len(x),))

        # Loop:
        for j in range(0, len(x)):
            xx = wk / (x[j] - xk)
            fx[j] = np.dot(xx, fvals) / xx.sum()
    else:
        # Loop over barycentric nodes
        # Initialise:
        num = np.zeros((len(x),))
        denom = np.zeros((len(x),))

        # Loop:
        for j in range(len(xk)):
            tmp = wk[j] / (x - xk[j])
            num = num + tmp * fvals[j]
            denom = denom + tmp

        fx = num / denom

    # Try to clean up NaNs:
    for k in np.nonzero(np.isnan(fx))[0]:
        index = np.nonzero(xk == x[k])[0]
        if len(index) > 0:
            fx[k] = fvals[index[0]]

    return fx


def barycentric_interpolation(x: np.array,
                              fvals: np.array,
                              xk: np.array,
                              wk: np.array = None) -> np.array:
    """ Barycentric interpolation formula
      barycentric_interpolation(x, fvals, xk, wk) uses the 2nd form barycentric formula with
      weights wk to evaluate an interpolant of the data {xk, fvals} at the points x.
      Note that xk, wk (if provided), and fk should all be arrays of the same length.

    :param x: a numpy array of points where we want to evaluate the interpolant
    :param fvals: a numpy array of values taken by the function at the points xk
    :param xk: points in [-1, 1]
    :param wk: the weights, if not given, they are computed using barycentric_weights()
    :return: a numpy array of the same length as x, containing values of the interpolant at x
    """

    # Make sure we have proper numpy arrays to begin with
    fvals = 1.0 * np.array(fvals)
    x = 1.0 * np.array(x)
    # Parse inputs:
    n = len(fvals)

    wk = wk or barycentric_weights(xk)

    assert len(xk) == len(wk), f'xk and wk must be of the same length: len(xk) = {len(xk)} but len(wk) = {len(wk)}'
    assert len(xk) == len(fvals), f'xk and fvals must be of the same length: len(xk) = {len(xk)} but len(fvals) = {len(fvals)}'

    # Trivial case
    if len(x) == 0:
        fx = x.copy()
        return fx

    # The function is a constant.
    if n == 1:
        fx = fvals * np.ones((len(x),))
        return fx

    # The function is NaN.
    if any(np.isnan(fvals)):
        fx = np.nan * np.ones((len(x),))
        return fx

    # The main loop:
    # Ignore divide by 0 warning:
    # [TODO] how to restore the warning state?
    np.seterr(divide='ignore', invalid='ignore')

    fx = barycentric_interpolation_jit(x, fvals, xk, wk)

    return fx


def barycentric_weights(x):
    """ Barycentric weights.
    returns scaled barycentric weights for the points in the
    array x. The weights w are scaled such that norm(w, inf) == 1.
    """
    # input dimension:
    n = len(x)

    # Capacity:
    if np.isreal(x).all():
        # capacity on a real interval
        capacity = 4.0 / (x.max() - x.min())
    else:
        # Scaling by capacity doesn't apply for complex nodes.
        capacity = 1.0

    # Compute the weights:
    w = np.ones(n)
    for j in range(0, n):
        v = capacity * (x[j] - x)
        v[j] = 1.0
        vv = np.exp(np.log(np.abs(v)).sum())
        w[j] = 1.0 / (np.prod(np.sign(v)) * vv)

    # Scaling:
    w = w / np.linalg.norm(w, np.inf)

    return w
