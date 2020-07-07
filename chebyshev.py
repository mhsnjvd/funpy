import numpy as np
from numba import njit
from barycentric import barycentric_interpolation
from functools import wraps


def complexify(f):
    """Decorator to apply f on real and imaginary parts and the return the sum
    :param f: A linear operator such that f(a+ib) = f(a) + i f(b)
    :return: a function which adds f(real(input)) + 1j * f(imag(input))
    """
    @wraps(f)
    def wrapper(c):
        """c is a complex input array"""
        # Make sure c is a numpy array
        c = 1.0 * np.array(c)
        if np.all(np.isreal(c)):
            return f(c.real)
        elif np.all(np.isreal(1j * c)):
            return 1j * f(c.imag)
        else:
            u = f(c.real)
            v = f(c.imag)
            return u + 1j * v

    return wrapper


@complexify
@njit
def chebyshev_coefficients_of_integral(coefficients: np.array) -> np.array:
    """Indefinite integral of a Function with given Chebyshev coefficients
    such that f(-1) = 0.

    NOTE: The algorithm works for complex coefficients, but for jit to
    work, we have to wrap this in the @complexify decorator

    ###########################################################################
    If the underlying function is represented as a n-vector c[k]:
              \sum_{k=0}^{n-1} c_k T_k(x)
    its integral is represented as a vector of length n+1 given by:
          \sum_{k=0}^{n} b_k T_k(x)
    where b_0 is determined from the constant of integration as
          b_0 = \sum_{k=1}^{n} (-1)^(k+1) b_k
    and other coefficients are given by
          b_1 = c_0 - c_2/2,
          b_k = (c_{k-1} - c_{k+1})/(2k) if 0 < k \leq n.
    with c_{n+1} = c_{n+2} = 0.

    Pages 32-33 of Mason & Handscomb,
    "Chebyshev Polynomials". Chapman & Hall/CRC (2003).
    ###########################################################################
    """

    # Handle the empty case:
    n = len(coefficients)
    if n == 0:
        return np.zeros((0,))

    # Make room in c[k] with zeros
    c = np.zeros((n + 2,))
    c[:n] = coefficients

    # Initialize vector b for the integral
    b = np.zeros((n + 1,))

    # values of b_(2) ... b_(n+1):
    b[2:] = (c[1:n] - c[3:n + 2]) / (2.0 * np.arange(2, n + 1))
    # value of b_1
    b[1] = c[0] - c[2] / 2.0
    v = np.ones((n,))
    v[1::2] = -1.0
    # value of b_0 such that f(-1) = 0
    b[0] = np.dot(v, b[1:])
    return b


@complexify
@njit
def chebyshev_definite_integral(coefficients: np.array) -> float:
    """ Definite integral of a function on the interval [-1, 1]."""

    n = len(coefficients)
    # Get the length of the coefficients:
    if n == 0:
        # Trivial cases:
        return np.nan
    elif n == 1:
        # Constant Function
        return 2.0 * coefficients[0]

    # General case
    c = np.zeros((n,))
    c[:n] = coefficients
    # Evaluate the integral using Chebyshev coefficients (see Thm. 19.2 of
    # Trefethen, Approximation Theory and Approximation Practice, SIAM, 2013, which
    # states that \int_{-1}^1 T_k(x) dx = 2/(1-k^2) for k even and zero for k odd).
    c[1::2] = 0.0

    # For jitted code, we have to do this slightly explicitly:
    d = np.zeros((n,))
    # k = 0 and k = 1 are handled separately
    d[:2] = [2.0, 0.0]
    d[2:n] = 2.0 / (1.0 - np.arange(2.0, n) ** 2)
    return np.dot(d, c)


@complexify
@njit
def chebyshev_coefficients_of_derivative(c: np.array) -> np.array:
    """Recurrence relation for coefficients of derivative.
    c is the array of coefficients of a Chebyshev series.
    c_out is the array of coefficients for the derivative.

    :param c: input coefficients
    :return: c_out: coefficients of the derivative
    """
    n = len(c)
    # Empty and constant case
    if n <= 1:
        return np.zeros((n,))

    c_out = np.zeros((n - 1,))
    w = 2.0 * np.arange(1.0, n)
    v = w * c[1:]
    c_out[n - 2::-2] = v[n - 2::-2].cumsum()
    c_out[n - 3::-2] = v[n - 3::-2].cumsum()
    c_out[0] = 0.5 * c_out[0]
    return c_out


def chebyshev_clenshaw_evaluation(x: np.array, coefficients: np.array) -> np.array:
    """A wrapper for chebyshev_clenshaw_evaluation_internal()
    """

    # Make sure x is cast to a numpy array
    x = 1.0 * np.array(x)

    # We only expect real x, so parametrise as a function of the coefficients c and
    # use the complexified version of the evaluation
    @complexify
    def g(c):
        return chebyshev_clenshaw_evaluation_internal(x, c)

    return g(coefficients)


@njit
def chebyshev_clenshaw_evaluation_internal(x: np.array, c: np.array) -> np.array:
    """Clenshaw's algorithm for evaluating a Chebyshev series with real
    coefficients c at points x.

    NOTE: the algorithm works for complex numbers, but since we are using jit, we restrict
    this to reals. One can remove @njit and use this code directly for the general case
    or use the wrapper chebyshev_clenshaw_evaluation() for general case.

    c is assumed to be an array of real numbers
    x is assumed to be an array
    y is an array of values of the Chebyshev expansion with coefficients c at x
    """

    # Clenshaw's algorithm for evaluating scalar-valued functions.
    bk1 = 0.0 * x
    bk2 = np.copy(bk1)
    x = 2.0 * x
    n = len(c)
    for k in np.arange(n - 1, 1, -2):
        bk2 = c[k] + x * bk1 - bk2
        bk1 = c[k - 1] + x * bk2 - bk1

    if not np.mod(n, 2):
        tmp = bk1
        bk1 = c[1] + x * bk1 - bk2
        bk2 = tmp

    y = c[0] + 0.5 * x * bk1 - bk2
    return y


@njit
def chebyshev_barycentric_weights(n: int) -> np.array:
    """Barycentric weights for Chebyshev points of 2nd kind.
    Returns n barycentric weights for polynomial interpolation on
    a Chebyshev grid of the 2nd kind. The weights are normalised so that they
    have infinity norm equal to 1 and the final entry is positive.

    See Thm. 5.2 of Trefethen, Approximation Theory and Approximation Practice,
    SIAM, 2013 for more information.

    :param n: an integer specifying the number of interpolation points
    :return: a numpy array of length n containing the weights
    """

    if n == 0:
        # Special case (no points)
        w = np.zeros((0,))
    elif n == 1:
        # Special case (single point)
        w = np.ones((1,))
    else:
        # General case
        w = np.ones(n)
        # The second last entry w[-2] and indexing backwards with a gap of 2 are all -1.0
        w[-2::-2] = -1.0
        # The last entry is always positive, and the first and the last entry are 0.5 in absolute value
        w[-1] = 0.5
        w[0] = .5 * w[0]
    return w


@njit
def chebyshev_points(n: int) -> np.array:
    """ Chebyshev points of 2nd kind in the interval [-1, 1]

    :param n: an non-negative integer
    :return: a numpy array of length n
    """
    if n == 0:
        # Special case (no points)
        x = np.zeros((0,))
    elif n == 1:
        # Special case (single point)
        x = np.zeros((1,))
    else:
        # Chebyshev points:
        m = n - 1.0
        # (Use of sine enforces symmetry.)
        x = np.sin(np.pi * (np.arange(-m, m + 1.0, 2.0) / (2.0 * m)))

    return x


def chebyshev_barycentric_interpolation(x: np.array, fvals: np.array) -> np.array:
    """Chebyhsev barycentric interpolation on a 2nd kind Chebyshev grid.
    The method evaluates f(x) using the barycentric interpolation formula, where f is the
    polynomial interpolant on a 2nd kind Chebyshev grid to the values passed in fvals

    Example:
    x = chebyshev_points(21);
    fx = 1.0 / (1.0 + 25.0 * x**2)
    xx = np.linspace(-1, 1, 1001)
    xx, yy = np.meshgrid(xx, xx)
    ff = chebyshev_barycentric_interpolation(xx + 1.0j * yy, fx)

    See also chebyshev_clesnshaw_evaluation()
    :param x: a numpy array
    :param fvals: values at a chebyshev grid of 2nd kind
    :return:
    """
    n = len(fvals)

    # Chebyshev points and Chebyshev barycentric weights:
    xk = chebyshev_points(n)
    wk = chebyshev_barycentric_weights(n)

    # Call the generic barycetnric interpolation code:
    fx = barycentric_interpolation(x, fvals, xk, wk)

    return fx


@njit
def angles_of_chebyshev_points(n: int) -> np.array:
    """Angles of Chebyshev points of 2nd kind in [-1, 1].
    """

    if n == 0:
        out = np.zeros((0,))
    elif n == 1:
        out = (np.pi / 2.0) * np.ones((1,))
    else:
        m = n - 1.0
        out = np.arange(m, -1.0, -1.0) * np.pi / m

    return out


def chebyshev_coefficients_to_values(chebyshev_coefficients: np.array) -> np.array:
    """ Convert Chebyshev coefficients to values at Chebyshev points of the 2nd kind.

    :param chebyshev_coefficients: a numpy array of n Chebyshev coefficients
    :return: a numpy array of length n with values at 2nd kind Chebyshev points
    """

    ################################################################################
    # [Developer Note]: This is equivalent to Discrete Cosine Transform of Type I.
    #
    # [Mathematical reference]: Sections 4.7 and 6.3 Mason & Handscomb, "Chebyshev
    # Polynomials". Chapman & Hall/CRC (2003).
    ################################################################################

    # *Note about symmetries* The code below takes steps to
    # ensure that the following symmetries are enforced:
    # even Chebyshev COEFFS exactly zero ==> VALUES are exactly odd
    # odd Chebychev COEFFS exactly zero ==> VALUES are exactly even
    # These corrections are required because the FFT used does not
    # guarantee that these symmetries are enforced.

    # This makes sure that input is cast to a float numpy array
    chebyshev_coefficients = 1.0 * np.array(chebyshev_coefficients)

    # Get the length of the input:
    n = len(chebyshev_coefficients)

    # Trivial case (constant or empty):
    if n <= 1:
        values = chebyshev_coefficients.copy()
        return values

    # check for symmetry
    is_even = np.max(np.abs(chebyshev_coefficients[1::2])) == 0.0
    is_odd = np.max(np.abs(chebyshev_coefficients[::2])) == 0.0

    # Scale the interior coefficients by 1/2:
    chebyshev_coefficients[1:n-1] = chebyshev_coefficients[1:n-1] / 2.0

    # Mirror the coefficients (to fake a DCT using an FFT):
    # Hack to make a complex array without specifying any data type
    tmp = 0j + np.zeros((n + (n - 2),))
    tmp[:n] = chebyshev_coefficients            # All coefficients
    tmp[n:] = chebyshev_coefficients[n-2:0:-1]  # mirrored without end points

    if np.all(np.isreal(chebyshev_coefficients)):
        # Real-valued case:
        values = np.fft.fft(tmp.real).real
    elif np.all(np.isreal(1j * chebyshev_coefficients)):
        # Imaginary-valued case:
        values = 1j * np.fft.fft(tmp.imag).real
    else:
        # General case:
        values = np.fft.fft(tmp)

    # Flip and truncate:
    values = values[n-1::-1]

    # [TODO] Is the np.fft already symmetric? in which
    # case we don't need this extra enforcing
    if is_even:
        values = (values + np.flipud(values)) / 2.0
    if is_odd:
        values = (values - np.flipud(values)) / 2.0

    return values


def chebyshev_values_to_coefficients(values: np.array) -> np.array:
    """ Convert values at Chebyshev points to Chebyshev coefficients.
      the method returns the (N+1)x1 vector C of coefficients such that F(x)
      = C(0)*T_0(x) + C(1)*T_1(x) + C(N)*T_N(x) (where T_k(x) denotes the
      k-th 1st-kind Chebyshev polynomial) interpolates the data [V_0, V_1, ..., V_N]
      at Chebyshev points of the 2nd kind.

      Input: values must be of type numpy.ndarray
      Output: a numpy.ndarray of the same size as values

    ################################################################################
    # This is equivalent to the Inverse Discrete Cosine Transform of Type I.
    #
    # [Mathematical reference]: Section 4.7 Mason & Handscomb, "Chebyshev
    # Polynomials". Chapman & Hall/CRC (2003).
    ################################################################################
    """

    # *Note about symmetries* The code below takes steps to
    # ensure that the following symmetries are enforced:
    # VALUES exactly even ==> odd Chebyshev COEFFS are exactly zero
    # VALUES exactly odd ==> even Chebyshev COEFFS are exactly zero
    # These corrections are required because the FFT used does not
    # guarantee that these symmetries are enforced.

    # This makes sure that input is cast to a float numpy array
    values = 1.0 * np.array(values)
    # Get the length of the input:
    n = len(values)

    # Trivial case (constant):
    if n <= 1:
        chebyshev_coefficients = values.copy()
        return chebyshev_coefficients

    # check for symmetry
    is_even = np.max(np.abs(values-np.flipud(values))) == 0.0
    is_odd = np.max(np.abs(values+np.flipud(values))) == 0.0

    # Mirror the values (to fake a DCT using an FFT):
    # Hack to make complex array with specifying a data type
    tmp = 0j + np.zeros((n - 1 + (n - 1),))
    tmp[:n-1] = values[n-1:0:-1]
    tmp[n-1:] = values[0:n-1]

    if np.isreal(values).all():
        # Real-valued case:
        chebyshev_coefficients = np.fft.ifft(tmp.real).real
    elif np.isreal(1j * values).all():
        # Imaginary-valued case:
        chebyshev_coefficients = 1j * np.fft.ifft(tmp.imag).real
    else:
        # General case:
        chebyshev_coefficients = np.fft.ifft(tmp)

    # Truncate:
    chebyshev_coefficients = chebyshev_coefficients[0:n]

    # Scale the interior coefficients:
    chebyshev_coefficients[1:n-1] = 2.0 * chebyshev_coefficients[1:n-1]

    # adjust coefficients for symmetry
    # [TODO] Is the np.fft already symmetric? in which
    # case we don't need this extra enforcing
    if is_even:
        chebyshev_coefficients[1::2] = 0.0
    if is_odd:
        chebyshev_coefficients[::2] = 0.0

    return chebyshev_coefficients


@njit
def chebyshev_quadrature_weights(n: int) -> np.array:
    """ Quadrature weights for Chebyshev points of 2nd kind.
    returns the N weights for Clenshaw-Curtis quadrature on 2nd-kind Chebyshev points.

    We use a variant of Waldvogel's algorithm [1], due to Nick Hale. (See below)

     Let $f(x) = \sum_{k=0}^nc_kT_k(x)$, then\vspace*{-3pt} }
       I(f) = v.'*c
     where
       v = \int_{-1}^1T_k(x)dx = { 2/(1-k^2) : k even
                                 { 0         : k odd
         = v'*inv(TT)*f(x) where TT_{j,k} = T_k(x_j)
         = (inv(TT)'*v)'*f(x)
     Therefore
       I(f) = w.'f(x) => w = inv(TT).'*v;
     Here inv(TT).' = inv(TT) is an inverse discrete cosine transform of Type I.

     Furthermore, since odd entries in v are zero, can compute via FFT without
     doubling up from N to 2N (though we still need to double up from N/2 to N to
     facilitate the use of ifft).

    References:
      1. Joerg Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
          quadrature rules", BIT Numerical Mathematics 46 (2006), pp 195-202.
      2. Greg von Winckel, "Fast Clenshaw-Curtis Quadrature",
          http://www.mathworks.com/matlabcentral/fileexchange/6911, (2005)
    """

    if n == 0:
        # Special case (no points!)
        out = np.zeros((0,))
    elif n == 1:
        # Special case (single point)
        out = 2 * np.ones((1,))
    else:
        # General case
        # Exact integrals of T_k (even)
        # c = 2/np.r_[1, 1-np.r_[2:n:2]**2]
        d = 1.0 - np.arange(2.0, n, 2.0)**2
        c = np.zeros((1 + len(d),))
        c[0] = 1.0
        c[1:] = d
        c = 2.0 / c

        # Mirror for DCT via FFT
        # c = np.r_[c, c[n//2-1:0:-1]]
        # w = np.fft.ifft(c).real
        c1 = c[n//2-1:0:-1]
        f = np.zeros((len(c) + len(c1),))
        f[:len(c)] = c
        f[len(c):] = c1
        w = np.fft.ifft(f).real
        # Boundary weights
        w[0] = w[0] / 2.0
        out = np.zeros(len(w) + 1)
        out[:len(w)] = w
        out[-1] = w[0]
    return out


@njit
def alias_chebyshev_coefficients(c: np.array, m: int) -> np.array:
    """Alias Chebyshev coefficients on the 2nd kind Chebyshev grid of length m.
    The method aliases the Chebyshev coefficients stored in the column vector coeffs
    to have length m. If m > len(c), the coefficients are padded with zeros.

    References:
      L.N. Trefethen, Approximation Theory and Approximation Practice, SIAM, 2013
      Page 27.

      Fox, L. and Parker, I. B., Chebyshev polynomials in Numerical Analysis,
      Oxford University Press, 1972.  (pp. 67)

      Mason, J. C. and Handscomb, D. C., Chebyshev polynomials, Chapman &
      Hall/CRC, Boca Raton, FL, 2003.  (pp. 153)
    """
    # Make sure everything is floating point:
    c = 1.0 * c

    n = len(c)

    aliased_coeffs = np.zeros((m,), dtype=c.dtype)

    if m > n:
        # Pad with zeros, just copy and return
        aliased_coeffs[:n] = c
        return aliased_coeffs

    # Alias coefficients: (see eq. (4.4) of Trefethen, Approximation Theory and
    # Approximation Practice, SIAM, 2013):

    if m == 0:
        return aliased_coeffs

    if m == 1:
        # Reduce to a single point:
        e = np.ones((n // 2 + n % 2,), dtype=c.dtype)
        e[1::2] = -1.0
        b = np.zeros((n // 2 + n % 2,), dtype=c.dtype)
        b[:len(b)] = c[::2]
        aliased_coeffs = np.array([np.dot(e, b)], dtype=c.dtype)
        return aliased_coeffs

    aliased_coeffs = np.copy(c)
    if m > n / 2:
        # If m > n/2, only single coefficients are aliased, and we can vectorise.
        j = np.arange(m, n)
        k = np.abs((j + m - 2) % (2 * m - 2) - m + 2)
        aliased_coeffs[k] = aliased_coeffs[k] + aliased_coeffs[j]
    else:
        # Otherwise loop. (slower)
        for j in np.arange(m, n):
            k = np.abs((j + m - 2) % (2 * m - 2) - m + 2)
            aliased_coeffs[k] = aliased_coeffs[k] + aliased_coeffs[j]

    # Truncate:
    aliased_coeffs = aliased_coeffs[:m]

    return aliased_coeffs


def chebyshev_to_monomial_coefficients(c: np.array):
    """Polynomial coefficients of a Chebyshev series
    returns coefficients in a vector a such that
     f(x) = a[n]*x^n + a[n-1]*x^(n-1) + ... + a[1]*x + a[0]
    reference: Mason & Handscomb, "Chebyshev Polynomials". Chapman & Hall/CRC (2003).
    """
    n = len(c)
    # Deal with empty case:
    if n == 0:
        return np.zeros((0,))

    # Coefficients on the unit interval:
    if n <= 2:
        # Constant and linear case:
        out = c
    else:
        # General case:

        # Flip in the beginning
        c = np.flipud(c)

        # Initialise working vectors:
        tn = np.zeros((n,))
        tnold1 = np.zeros((n,))
        tnold1[1] = 1.0
        tnold2 = np.zeros((n,))
        tnold2[0] = 1.0

        # Initialise output:
        out = np.zeros((n,))

        # Initial step:
        out[[0, 1]] = np.array([0.0, c[-1] * tnold2[0]]) + (c[-2] * tnold1[[1, 0]])

        # Recurrence:
        for k in range(2, n):
            tn[:k + 1] = np.r_[0.0, 2.0 * tnold1[:k]] - np.r_[tnold2[:k - 1], 0.0, 0.0]
            out[:k + 1] = c[-k - 1] * np.flipud(tn[:k + 1]) + np.r_[0.0, out[:k]]
            # It is important to copy here:
            # [TODO]: tnold1[:] fails, investigate why we need explicit copy here
            tnold2 = np.copy(tnold1)
            tnold1 = np.copy(tn)

    return out[::-1]




