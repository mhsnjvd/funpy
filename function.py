import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import copy
import types
from collections.abc import Iterable
from functools import wraps
from chebyshev import chebyshev_values_to_coefficients
from chebyshev import chebyshev_coefficients_to_values
from chebyshev import chebyshev_barycentric_interpolation
from chebyshev import chebyshev_coefficients_of_derivative
from chebyshev import chebyshev_points
from chebyshev import chebyshev_clenshaw_evaluation
from chebyshev import chebyshev_definite_integral
from chebyshev import chebyshev_coefficients_of_integral
from chebyshev import chebyshev_to_monomial_coefficients


def scalar_vector_mix(func):
    """Decorator which returns scalars for 
    scalars and vectors for vectors. Assumes
    that f can only accept and return vector
    :param func: function to be decorated
    :return: the decorated function
    """
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        x_is_scalar = not isinstance(x, Iterable)
        if x_is_scalar:
            # Cast to an array and then numpy array
            xx = 1.0 * np.array([x])
        else:
            # Cast to numpy array, just to allow calls like func([-1, 0, 1]) etc
            xx = 1.0 * np.array(x)

        fx = func(xx, *args, **kwargs)

        if x_is_scalar:
            assert len(fx) == 1, f'scalar valued evaluation must return array of length 1, returned {fx}'
            fx = fx[0]
        return fx
    return wrapper


class Fun:
    """Class for function approximations on the interval [-1, 1]"""
    # Initialize properties of the object
    __default_dtype__ = np.complex128
    __default_tolerance__ = np.spacing(1)
    __default_min_samples__ = 2**3 + 1
    __default_max_samples__ = 2**16 + 1

    @property
    def minimum_samples(self):
        return Fun.__default_min_samples__

    @property
    def maximum_samples(self):
        return Fun.__default_max_samples__

    @property
    def tolerance(self):
        return Fun.__default_tolerance__

    @property
    def domain(self):
        if self.length > 0:
            return np.array([-1.0, 1.0])
        else:
            # TODO: Empty function should have no domain? or
            return np.zeros((0,))

    # @domain.setter
    # def domain(self, dom: np.array):
    #     # We provide this method so that the variable .__domain exists, it
    #     # can only by empty in case of an empty object, in all other cases
    #     # it is forced to only accept the value [-1, 1]
    #     assert len(dom) == 2 and np.all(dom == np.array([-1.0, 1.0]))
    #     self.__domain = np.array(dom)
    #     return

    @property
    def values(self):
        return self.coefficients_to_values()

    @property
    def length(self):
        return len(self.coefficients)

    @property
    def coefficients(self) -> np.array:
        # Always return a copy to make sure it is not written to
        return self.__coefficients.copy()

    @coefficients.setter
    def coefficients(self, coeffs: np.array):
        assert isinstance(coeffs, type(np.zeros(0,))), f'coefficients must be 1-dimensional numpy array, type of input coefficients is {type(coeffs)}'
        assert len(coeffs.shape) == 1, f'coefficients must be 1-dimensional numpy array, shape of input coefficients is {coeffs.shape}'
        self.__coefficients = coeffs.copy()

    @property
    def points(self):
        return chebyshev_points(self.length)

    # TODO: work on this to allow construction from arbitrary points
    # @points.setter
    # def points(self, pts: np.array):
    #     assert isinstance(pts, type(np.zeros(0,))), f'points must be 1-dimensional numpy array, type of input points is {type(pts)}'
    #     assert len(pts.shape) == 1, f'points must be 1-dimensional numpy array, shape of input points is {pts.shape}'
    #     self.__points = pts

    @property
    def resolved(self):
        return self.__resolved

    @resolved.setter
    def resolved(self, status: bool):
        self.__resolved = status

    @property
    def function(self):
        return self.__function

    @function.setter
    def function(self, fun):
        self.__function = fun
        return

    def construct_from_lambda(self, fun, length=None):
        """Adaptive or fixed length construction of a function object from lambda"""
        self.function = fun

        # TODO: Do we want this:
        # Handle construction by string:
        # if isinstance(fun, str):
        #     # Try to convert it to a lambda (naive)
        #     op = eval('lambda x: ' + op)

        ######################## Non Adaptive construction. ###################
        if length is not None:
            assert (length == np.round(length)) and length >= 0, f'length must be a non-negative int, input length = {length}'
            if length == 0:
                # Special case
                self.coefficients = np.zeros((0,))
            else:
                x = chebyshev_points(length)
                # Evaluate the function at x, and convert these values to coefficients
                self.coefficients = self.values_to_coefficients(fun(x))

            self.resolved = True
            return
        ######################### Adaptive construction. ######################
        # Initialise empty values to pass to refine:
        values = None
        v_scale = 0.0
        h_scale = 1.0

        # Loop until resolved or give_up:
        while True:
            # Call the appropriate refinement routine:
            values, give_up = self.refine_by_nesting(fun, values)

            # We are giving up, oh no :(
            if give_up:
                self.resolved = False
                self.coefficients = self.values_to_coefficients(values)
                print(f'Function did not converge after {len(values)} samples')
                return

            # Update vertical scale: (Only include sampled finite values)
            values_temp = np.copy(values)
            values_temp[~np.isfinite(values)] = 0.0
            v_scale = np.max([v_scale, np.max(np.abs(values_temp))])

            # Compute the Chebyshev coefficients:
            self.coefficients = self.values_to_coefficients(values)

            # Check for resolution:
            resolved, cutoff = self.standard_check(values, v_scale, h_scale)

            if resolved:
                # discard unwanted coefficients
                self.coefficients = self.prolong_coefficients(self.coefficients, cutoff)
                self.resolved = resolved
                # All done here, return
                return

    def __init__(self,
                 fun=None,                               # A lambda to construct a function
                 length=None,                            # construct fun on a grid of this length
                 values=np.zeros((0,)),                  # Values taken by the function
                 coefficients=np.zeros((0,)),            # Chebyshev coefficients of the function
                 points=np.zeros((0,)),                  # Points on which values are taken, default to Chebyshev
                 ):
        """To construct a function, you can

        1. Pass a function i.e., a lambda in which case values,
        coefficients, and points should not be passed in or should
        be equal to their default values

        This triggers the automatic construction of a Chebyshev
        polynomial interpolant.

        2. Pass in coefficients, in which case all other arguments should
        be passed in at their default values or not passed at all

        3. Pass in a set of points and values (both arrays of equal lengths)
        in which case all other arguments should be at their default values
        or should not be passed in at all.

        :param fun:
        :param values:
        :param coefficients:
        :param points:
        """
        # Empty Function case:
        if (fun is None) and (length is None) and (len(coefficients) == 0) and (len(values) == 0):
            assert len(points) == 0, f'no values passed, can not pass points, points passed = {points}'
            self.coefficients = coefficients
            # An empty function is resolved by definition
            self.resolved = True

        if fun is not None:
            assert len(coefficients) == 0, f'function passed, coefficents must be empty, however, coefficents = {coefficients}'
            assert len(values) == 0, f'function passed, values must be empty, however, input_values = {values}'
            assert len(points) == 0, f'function passed, points must be empty, however, input_points = {points}'

            self.construct_from_lambda(fun=fun, length=length)
        else:
            assert length is None, f'length can only be used if a fun is passed. Input length = {length}'

        if len(coefficients) > 0:
            assert fun is None, f'coefficients passed, coefficients must be empty, however, coefficients = {coefficients}'
            assert len(values) == 0, f'coefficients passed, values must be empty, however, input_values = {values}'
            assert len(points) == 0, f'coefficients passed, points must be empty, however, input_points = {points}'
            self.coefficients = coefficients
            self.resolved = True

        if len(values) > 0:
            if len(points) == 0:
                # Assume values at chebyshev points
                self.coefficients = chebyshev_values_to_coefficients(values)
                # The evaluation function is available via the barycentric interpolant
                self.function = lambda x: chebyshev_barycentric_interpolation(x, values)
                self.resolved = True
            else:
                assert len(values) == len(points), f'values and points must have same length'
                # TODO: not implemented yet
        elif len(values) == 0:
            assert len(points) == 0, f'no values passed, can not pass points, points passed = {points}'

    def copy(self):
        """Deep copy of a function
        :return: an object of type function
        """
        return copy.deepcopy(self)

    def coefficients_to_values(self, coeffs=None):
        if coeffs is None:
            coeffs = self.coefficients
        return chebyshev_coefficients_to_values(coeffs.copy())

    def values_to_coefficients(self, values=None):
        if values is None:
            values = self.coefficients_to_values()
        return chebyshev_values_to_coefficients(values.copy())

    #############################################################
    ######## section: plotting for Fun class          ###########
    #############################################################
    def plot(self, *args, **kwargs):
        a, b = self.domain[0], self.domain[-1]
        x = np.linspace(a, b, 2001)
        y = self(x)
        if not np.all(np.isreal(y)):
            print('Discarding imaginary values in plot')
        y = y.real
        ax = plt.plot(x, y, *args, **kwargs)
        plt.grid(True)
        plt.show()
        return ax

    def plot_coefficients(self, *args, **kwargs):
        """Display Chebyshev coefficients.
        plots the Chebyshev coefficients of a Function f on a semilogy scale.
        
        Note: to make the log of coefficients comparable, zero coefficients have a small
        value added to them (typically close to relative machine precision)).
        """

        # Some argument parsing, those that are not understood by plt.plot but
        # we might want to have them:
        loglog_plot = kwargs.pop('loglog', False)

        # Deal with an empty input:
        if self.length == 0:
            return

        # The coefficients and vertical scale:
        abs_coeffs = np.abs(self.coefficients)
        v_scale = self.vscale()

        # Add a tiny amount to zeros to make plots look nicer:
        if v_scale > 0:
            # Min of eps*vscale and the minimum non-zero coefficient:
            min_nonzero_coeff = np.min(abs_coeffs[np.nonzero(abs_coeffs)])
            abs_coeffs[abs_coeffs == 0.0] = np.min(np.r_[np.spacing(1) * v_scale, min_nonzero_coeff])
        else:
            # Add machine eps for zero functions:
            abs_coeffs = abs_coeffs + np.spacing(1)

        # Get the size:
        n = len(abs_coeffs)

        xx = np.arange(0.0, n)
        yy = abs_coeffs

        # Plot the coefficients:
        if loglog_plot:
            plt.loglog(xx, yy, *args, **kwargs)
            min_xlim = 0.8
        else:
            plt.semilogy(xx, yy, *args, **kwargs)
            min_xlim = 0.0

        # By default, set grid on
        plt.grid(True)

        current_axis = plt.gca()

        # Adjust x-limit:
        x_lim = current_axis.get_xlim()
        current_axis.set_xlim([np.min([x_lim[0], min_xlim]), np.max([x_lim[1], n])])

        # Add title and labels
        current_axis.set_title('Chebyshev coefficients')
        current_axis.set_xlabel('Degree of Chebyshev polynomial')
        current_axis.set_ylabel('Magnitude of coefficient')

        return current_axis

    ############################################################
    ##########   Boolearn Operators          ###################
    ############################################################
    def __eq__(self, other):
        return self.isequal(other)

    def any(self):
        return np.any(self.coefficients)

    def all(self):
        return len(self.roots()) == 0

    def isreal(self):
        """True for a real function"""
        # Check if all the coefficients are real:
        return np.all(np.isreal(self.coefficients))

    def iszero(self):
        """True for zero Function objects. """
        # TODO: supply tolerances here?
        return np.allclose(self.coefficients, 0.0 * self.coefficients)

    def isequal(self, other):
        # [TODO] Only coefficients are checked
        # resolution is not compared
        if not isinstance(other, Fun):
            # [TODO] something must be done
            print('isequal() accepts a Fun object only.')

        # Get coefficients and trim zeros at the end
        a = np.trim_zeros(self.coefficients, 'b')
        b = np.trim_zeros(self.coefficients, 'b')

        # compare coefficients
        # TODO: this is quite weired, should we still use np.allclose?
        # np.allclose([0], [0, 0, 0])
        # Out[42]: True
        # np.allclose([1], [0, 0, 0])
        # Out[43]: False
        # TODO: also provide tolerances
        return np.allclose(a, b)

    def isfinite(self):
        """Test if a Function is bounded.
        returns False if f has any infinite coefficients
        """
        # Check if coefficients are finite:
        return np.all(np.isfinite(self.coefficients))

    def isinf(self):
        """Test if a Function is unbounded.
        returns True if f has any infinite coefficients
        """
        # Check if any coefficients are infinite:
        return np.any(np.isinf(self.coefficients))

    def isnan(self):
        """Test if a Function has any NaN coefficients"""
        # Check if any coefficients are NaN:
        return np.any(np.isnan(self.coefficients))

    ############################################################
    #######  section: operators that output a Fun ##############
    ############################################################
    # NOTE:
    # These methods must generate the output by copying one of the
    # inputs to make sure that all classes which inherit from
    # this class work properly. (e.g., domain information
    # from the subclass "Function" might get destroyed if we
    # do not copy the input
    ############################################################
    def prolong(self, n: int):
        result = self.copy()
        result.coefficients = self.prolong_coefficients(self.coefficients, n)
        return result

    def real(self):
        """Real part of a Fun."""
        result = self.copy()
        result.coefficients = result.coefficients.real
        return result

        # # Compute the real part of the coefficients:
        # c = self.coefficients.real.copy()
        #
        # # TODO: what atol and reltol should be used below?
        # if np.allclose(c, np.zeros((len(c),)), atol=self.tolerance):
        #     # Input was purely imaginary, so output a zero Function:
        #     return Fun(coefficients=np.zeros(1, dtype=c.dtype))
        # else:
        #     return Fun(coefficients=c)

    def imag(self):
        """Imaginary part of a Fun."""
        return (-1.0j * self).real()

    def conjugate(self):
        """Conjugate of a Fun."""
        out = self.copy()
        if not self.isreal():
            out.coefficients = np.conjugate(out.coefficients)
        return out

    def conj(self):
        """Alias of conjugate"""
        return self.conjugate()

    def fix(self):
        out = self.copy()
        # Evaluate at the two end points, and an arbitrary interior point:
        arbitraryPoint = 0.1273123881594
        fx = self([-1.0, arbitraryPoint, 1.0])
        # Take the mean:
        meanfx = np.mean(fx)
        # Compute the fix:
        out.coefficients = np.fix(meanfx)
        return out

    def fliplr(self):
        """Has no effect on the object."""
        return self.copy()

    def flipud(self):
        """Flip/reverse a function object.
          returns g such that g(x) = f(-x) for all x in [-1, 1].
        """
        out = self.copy()
        # Negate the odd coefficients:
        out.coefficients[1::2] = -out.coefficients[1::2]
        return out

    def floor(self):
        out = self.copy()
        # Evaluate at the two end points, and an arbitrary interior point:
        arbitrary_point = 0.1273123881594
        fx = self([-1.0, arbitrary_point, 1.0])
        # Take the mean:
        mean_fx = np.mean(fx)
        # Compute the fix:
        out.coefficients = np.floor(mean_fx)
        return out

    def abs(self):
        """Absolute value of a function.
         returns the absolute value of f, where f has no roots in [-1 1].
         WARNING: If len(f.roots()) > 0, f.abs() will return garbage
         with no warning. f can be complex.
        """
        # Make a copy in the beginning to make sure
        # all info is saved for child class instances as well
        out = self.copy()
        if self.isreal() or (1.0j * self).isreal():
            # Convert to values and then compute the abs():
            abs_f = Fun(values=np.abs(self.values))
        else:
            # [TODO]
            # f = compose(f, @abs, [], [], varargin{:});
            abs_f = Fun(lambda x: np.abs(self(x)))

        # copy the coefficients
        out.coefficients = abs_f.coefficients.copy()
        return out

    def cumsum(self):
        """Indefinite integral of a Fun."""
        return self.integral()

    def integral(self):
        """Indefinite integral of a Fun."""
        f = self.copy()
        f.coefficients = chebyshev_coefficients_of_integral(self.coefficients.copy())

        # [TODO] should we simplify or not?
        # f = f.simplify()

        # Ensure f(-1) = 0:
        # TODO: We should adjust for 'lval' etc in the future?
        # c = f.coefficients.copy()
        # c[0] = c[0] - f.lval()
        # Individual coefficients can not be written to directly, so we have to
        # do the above copying first
        # f.coefficients = c
        return f

    def diff(self, order: int = 1):
        return self.derivative(order)

    def derivative(self, order: int = 1):
        """Derivative of a function
        f.derivative(k) is the kth derivative.

        ################################################################################
        If the Function f of length n is represented as
              \sum_{k=0}^{n-1} a_k T_k(x)
        its derivative is represented with coefficients of length n-1 given by
              \sum_{k=0}^{n-2} c_k T_k(x)
        where c_0 is determined by
              c_0 = c_2/2 + a_1;
        and for k > 0,
              c_k = c_{k+2} + 2 * (k+1) * a_{k+1},
        with c_n = c_{n+1} = 0.

        [Reference]: Mason & Handscomb, "Chebyshev Polynomials". Chapman & Hall/CRC (2003).
        ################################################################################
        """

        assert (order == np.round(order) and order >= 0), f'order must be a non-negative integer, input order = {order}'

        # Make a copy for the derivative:
        fp = self.copy()

        n = len(self)
        # Trivial case of an empty Function:
        if n == 0 or order == 0:
            return fp

        # Get the coefficients:
        c = np.copy(self.coefficients)

        # If k >= n, we know the result will be the zero function:
        if order >= n:
            fp.coefficients = np.array([0.0])

        # Loop for higher derivatives:
        for m in range(order):
            # Compute new coefficients using recurrence:
            c = chebyshev_coefficients_of_derivative(c)

        # Return a function made of the new coefficients:
        fp.coefficients = c
        return fp

    ############################################################
    ###### section: operator overloads that output a Fun #######
    ############################################################
    def __pos__(self):
        # TODO: to copy or not ot copy here?
        return self.copy()

    def __neg__(self):
        result = self.copy()
        result.coefficients = -1.0 * result.coefficients
        return result

    def __add__(self, other):
        if not isinstance(other, Fun):
            return self.__radd__(other)

        n = self.length
        m = other.length

        result = self.copy()

        # Check for the empty case
        if n == 0 or m == 0:
            result.coefficients = np.zeros((0,))
            return result

        if n >= m:
            coeffs = np.r_[other.coefficients, np.zeros((n-m,))]
            result.coefficients = self.coefficients + coeffs
        else:
            coeffs = np.r_[self.coefficients, np.zeros((m-n,))]
            result.coefficients = other.coefficients + coeffs

        result.resolved = self.resolved and other.resolved

        return result

    def __radd__(self, other):
        result = self.copy()
        if self.length > 0:
            # CAUTION: Coefficients can not be operated upon entry by entry, we
            # must create the coefficients separately, work on them and then assign
            # This is because .coefficients is a _property_ and statements like
            # self.coefficients[0] = self.coefficients[0] + 2 have not effect
            c = result.coefficients.copy()
            if not np.isreal(other):
                c = c + 0.0j

            c[0] = c[0] + other
            result.coefficients = c

        return result

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return self.__radd__(-1.0 * other)

    def __mul__(self, other):
        if not isinstance(other, Fun):
            return self.__rmul__(other)

        n = len(self.coefficients)
        m = len(other.coefficients)

        result = self.copy()
        if (n == 0) or (m == 0):
            # Empty cases
            result.coefficients = np.zeros((0,))
            return result
        elif n == 1:
            # Constant case
            return other.__rmul__(self.coefficients[0])
        elif m == 1:
            # Constant case
            return self.__rmul__(other.coefficients[0])
        else:
            # General case
            fc = np.r_[self.coefficients[:],  np.zeros((m+1,))]
            gc = np.r_[other.coefficients[:], np.zeros((n+1,))]

            # N = m + n + 1
            N = len(fc)
            # Toeplitz vector.
            t = np.r_[2.0 * fc[0], fc[1:]]
            # Embed in Circulant.
            x = np.r_[2.0*gc[0], gc[1:]]
            # FFT for Circulant mult.
            xprime = np.fft.fft(np.r_[x, x[-1:0:-1]])
            aprime = np.fft.fft(np.r_[t, t[-1:0:-1]])
            # Diag in function space.
            Tfq = np.fft.ifft(aprime * xprime)
            out_coeffs = 0.25 * np.r_[Tfq[0], Tfq[1:] + Tfq[-1:0:-1]]
            out_coeffs = out_coeffs[:N]

            result.coefficients = out_coeffs

            # Check for two cases where the output is known in advance to be positive,
            # namely f == conj(g) or (f == g and isreal(f)).
            # TODO: maybe we should switch from np.array_equal() to np.all_close()
            result_is_positive = ((np.array_equal(self.coefficients, other.coefficients) and self.isreal())
                                  or (np.array_equal(np.conjugate(self.coefficients), other.coefficients)))

            #% [TODO] Update resolved:
            #f.resolved = f.resolved and g.resolved

            # we simplify here:
            result = result.simplify()

            # Make sure real inputs give real output:
            # TODO: force real/imag on other cases as well
            if self.isreal() and other.isreal():
                result = result.real()

            if result_is_positive:
                # Here we know that the product of f and g should be positive. However,
                # simplify might have destroyed this property, so we enforce it.
                values = self.coefficients_to_values(result.coefficients)
                result.coefficients = self.values_to_coefficients(np.abs(values))
            return result

    def __rmul__(self, other):
        result = self.copy()
        if self.length > 0:
            result.coefficients = self.coefficients * other

        return result

    def __pow__(self, a):
        """Function raised to the power a."""
        result = self.copy()
        if self.length > 0:
            f_pow_a = Fun(lambda x: self(x)**a)
            result.coefficients = f_pow_a.coefficients.copy()
        return result

    ############################################################
    ##########   section: misc overloads               #########
    ############################################################
    def __len__(self):
        return self.length

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = f'Object of class Fun of length {self.length} on {self.domain}\n'
        #if self.fun:
        #    fun_str = inspect.getsource(self.fun).split('=')[1]
        #    # Remove \n character at the end
        #    fun_str = fun_str[:-1]
        #    s = s + "Constructed via: %s" % fun_str
        return s

    def __call__(self, x):
        if isinstance(x, types.LambdaType):
            # A lambda has been passed
            f = x
            # square brackets are crucial to avoid infinite recursion See __getitem__
            return Fun(lambda t: self[f(t)])
        elif isinstance(x, Fun):
            # A Function has been passed
            f = x
            # Use square brackets in both cases
            # TODO: how do we ensure domain when this is called from subclass
            return Fun(lambda t: self[f[t]])
        else:
            # square brackets are crucial to avoid infinite recursion See __getitem__
            return self[x]

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        return self.evaluate(x)

    ############################################################
    ##########   section: methods that output numbers  #########
    ############################################################
    # TODO: try using @scalar_vector_mix decorator
    def evaluate(self, x):
        """Evaluate the object at the point(s) x"""
        
        x_is_scalar = not isinstance(x, Iterable)

        if x_is_scalar:
            # Cast to an array and then numpy array
            xx = 1.0 * np.array([x])
        else:
            # Cast to numpy array, just to allow calls like f([-1, 0, 1]) etc
            xx = 1.0 * np.array(x)

        # Main call to the evaluation algorithm:
        fx = chebyshev_clenshaw_evaluation(xx, self.coefficients)

        if x_is_scalar:
            assert len(fx) == 1, f'scalar valued evaluation must return array of length 1, returned {fx}'
            fx = fx[0]

        return fx

    def sum(self):
        """Definite integral of a function on the interval [-1, 1]."""
        return self.definite_integral()

    def definite_integral(self):
        """Definite integral of a function on the interval [-1, 1]."""
        return chebyshev_definite_integral(self.coefficients)

    def poly(self):
        """Polynomial coefficients of a function.
        c = poly(f) returns the polynomial coefficients of f so that
        f(x) = c[n]*x^n + C[n-1]*x^(n-1) + ... + c[1]*x + c[0]
        """
        # Deal with empty case:
        if self.length == 0:
            return np.zeros((0,))

        return chebyshev_to_monomial_coefficients(self.coefficients)

    def vscale(self):
        if self.length == 0:
            return np.nan
        else:
            return np.max(np.abs(self.values))

    def minandmax(self):
        """Global minimum and maximum on [-1,1].
        returns a tuple (vals, pos).
        vals is a numpy array of length 2: [min(f), max(f)] with the
        global minimum and maximum of the f on [-1,1].

        pos is a numpy array of length 2: [argmin(f), argmax(f)]

        If f is complex-valued the absolute values are taken to determine extrema
        but the resulting values correspond to those of the original function. That
        is, vales = f(pos) where _, pos = f.abs().minandmax(). (The algorithm
        actually computes (f.abs()**2).minandmax() to avoid introducing
        singularities in the function).

        """

        if not self.isreal():
            # We compute extrame of |f|^2 to avoid introducing a singularity.
            realf = self.real()
            imagf = self.imag()
            h = realf * realf + imagf * imagf
            h = h.simplify()
            ignored, pos = h.minandmax()
            # Return values of the original function
            vals = self[pos]
            return vals, pos

        # Compute derivative:
        fp = self.derivative()

        # Make the Chebyshev grid (used in minandmax).
        xpts = self.points

        # Initialise output
        pos = np.zeros((2,))
        vals = np.zeros((2,))

        # Constant function
        if self.length == 1:
            vals = self[pos]
            return vals, pos

        # Compute critical points:
        r = fp.roots()
        r = np.unique(np.r_[-1.0, r, 1.0])
        v = self[r]

        # min
        vals[0] = np.min(v)
        pos[0] = r[np.argmin(v)]

        # Take the minimum of the computed minimum and the function values:
        values = self.coefficients_to_values(self.coefficients)
        temp = np.r_[vals[0], values]
        vmin = np.min(temp)
        vindex = np.argmin(temp)
        if vmin < vals[0]:
            vals[0] = vmin
            pos[0] = xpts[vindex - 1]

        # max
        vals[1] = np.max(v)
        pos[1] = r[np.argmax(v)]

        # Take the maximum of the computed maximum and the function values:
        temp = np.r_[vals[1], values]
        vmax = np.max(temp)
        vindex = np.argmax(temp)
        if vmax > vals[1]:
            vals[1] = vmax
            pos[1] = xpts[vindex - 1]

        return vals, pos

    def max(self):
        """Global maximum on [-1,1]."""
        minmax, pos = self.minandmax()
        return minmax[1]

    def argmax(self):
        """Location of global maximum on [-1,1]."""
        minmax, pos = self.minandmax()
        return pos[1]

    def min(self):
        """Global minimum on [-1,1]."""
        minmax, pos = self.minandmax()
        return minmax[0]

    def argmin(self):
        """Location of global minimum on [-1,1]."""
        minmax, pos = self.minandmax()
        return pos[0]

    def roots(self, **kwargs):
        """Roots of a function in the interval [-1,1]."""

        def roots_main(c, htol):
            """Compute the roots of the polynomial given by the coefficients c on
            the unit interval."""
       
            # Simplify the coefficients:
            tail_max = np.spacing(1) * np.abs(c).sum()
            # Find the final coefficient close to tail_max:
            big_coeffs_mask = np.where(np.abs(c) > tail_max)[0]
            if len(big_coeffs_mask) == 0:
                n = 0
            else:
                n = big_coeffs_mask[-1] + 1

            # Truncate the coefficients:
            if 1 < n < len(c):
                c = c[:n]
       
            max_eig_size = 50

            if n == 0:
                if roots_pref['zero_fun']:
                    r = np.zeros((1,))
                else:
                    r = np.zeros((0,))
            elif n == 1:
                if c[0] == 0.0 and roots_pref['zero_fun']:
                    r = np.zeros((1,))
                else:
                    r = np.zeros((0,))
            elif n == 2:
                r = -c[0] / c[1]
                if not roots_pref['all']:
                    if np.abs(r.imag) > htol or r < -(1 + htol) or r > (1 + htol):
                        r = np.zeros((0,))
                    else:
                        r = np.min([r.real, 1])
                        r = np.max([r, -1])
            elif not roots_pref['recurse'] or n <= max_eig_size:
                c_old = np.copy(c)
                c = -0.5 * c[:-1] / c[-1]
                c[-2] = c[-2] + 0.5

                oh = 0.5 * np.ones(len(c) - 1, dtype=c.dtype)
                A = np.diag(oh, 1) + np.diag(oh, -1)
                A[-2, -1] = 1.0
                A[:, 0] = np.flipud(c)

                if roots_pref['qz']: 
                    B = np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
                    # c_old will be complex or real based on values in B
                    c_old = np.array(c_old / np.abs(c_old).max(), dtype=B.dtype)
                    B[0, 0] = c_old[-1]
                    c_old = -0.5 * c_old[:-1]
                    c_old[-2] = c_old[-2] + 0.5 * B[0, 0]
                    A[:, 0] = np.flipud(c_old)
                    r = eig(A, b=B)[0]
                else:
                    r = eig(A)[0]

                # Clean the roots up a bit:
                if not roots_pref['all']: 
                    # Remove dangling imaginary parts:
                    mask = np.abs(r.imag) < htol
                    r = r[mask].real
                    # Keep roots inside [-1 1]:
                    r = np.sort(r[np.abs(r) <= 1 + htol])
                    # Correct roots over ends:
                    if len(r) != 0:
                        r[0] = np.max([r[0], -1])
                        r[-1] = np.min([r[-1], 1])
                elif roots_pref['prune']:
                    rho = np.sqrt(np.spacing(1))**(-1/n)
                    rho_roots = np.abs(r + np.sqrt(r**2 - 1))
                    rho_roots[rho_roots < 1] = 1.0/rho_roots[rho_roots < 1]
                    r = r[rho_roots <= rho]
            # Otherwise, split using more traditional methods (i.e., Clenshaw):
            else:
                # Evaluate the polynomial on both intervals:
                x_left = chebptsAB(n, [-1.0, split_point])
                x_right = chebptsAB(n, [split_point, 1])
                xx = np.r_[x_left, x_right]
                v = chebyshev_clenshaw_evaluation(xx, c)
       
                # Get the coefficients on the left:
                c_left = chebyshev_values_to_coefficients(v[:n])
       
                # Get the coefficients on the right:
                c_right = chebyshev_values_to_coefficients(v[n:])
       
                # Recurse:
                r_left = roots_main(c_left, 2*htol)
                r_right = roots_main(c_right, 2*htol)
                r1 = (split_point - 1.0)/2.0 + (split_point + 1.0)/2.0 * r_left
                r2 = (split_point + 1.0)/2.0 + (1.0 - split_point)/2.0 * r_right
                r = np.r_[r1, r2]

            return np.sort(np.array(r).flatten())

        def chebptsAB(n, ab):
            """chebpts in an interval."""
            a = ab[0]
            b = ab[1]
            x = chebyshev_points(n)
            y = b * (x + 1.0) / 2.0 + a * (1.0 - x) / 2.0
            return y

        # Deal with the empty case:
        if len(self.coefficients) == 0:
            return np.zeros((0,))
        
        # Default preferences:
        roots_pref = {'all': kwargs.setdefault('all', False),
                      'complex_roots': kwargs.setdefault('complex_roots', False),
                      'recurse': kwargs.setdefault('recurse', True),
                      'prune': kwargs.setdefault('prune', False),
                      'zero_fun': kwargs.setdefault('zero_fun', True),
                      'qz': kwargs.setdefault('qz', False),
                      'filter': kwargs.setdefault('filter', None)}

        if roots_pref['complex_roots']:
            roots_pref['all'] = True
            roots_pref['prune'] = True

        # Subdivision maps [-1,1] into [-1, split_point] and [split_point, 1].
        # This is an arbitrary number.
        split_point = -0.005847724917629

        # Trivial case for f constant:
        if self.length == 1 or self.vscale() == 0.0:
            if self.coefficients[0] == 0.0 and roots_pref['zero_fun']:
                # Return a root at centre of domain:
                out = np.array([0.0])
            else:
                # Return empty:
                out = np.zeros((0,))
            return out

        # Get scaled coefficients for the recursive call:
        c = np.copy(self.coefficients) / self.vscale()

        # Call the recursive roots function:
        # TODO:  Does the tolerance need to depend on some notion of hscale?
        default_tol = np.spacing(1) * 100.0
        r = roots_main(c, default_tol)

        # [TODO] Try to filter out spurious roots:
        if roots_pref['filter'] is not None:
            # r = np.sort(r, 'ascend');
            # r = np.sort(r, 'ascend');
            # fltr = roots_pref['filter']
            # r = fltr(r, f)
            pass

        # Prune the roots, if required:
        if roots_pref['prune'] and not roots_pref['recurse']:
            rho = np.sqrt(np.spacing(1))**(-1.0 / self.length)
            rho_roots = np.abs(r + np.sqrt(r**2 - 1.0))
            rho_roots[rho_roots < 1.0] = 1.0 / rho_roots[rho_roots < 1.0]
            out = r[rho_roots <= rho]
        else:
            out = r

        return out

    ############################################################
    ##########   section: constructor methods  #################
    ############################################################
    def prolong_coefficients(self, c: np.array, n: int) -> np.array:
        """Manually adjust the number of points used in a Fun.
          prolong_coefficients(c, n) returns d where len(d) = n and d represents
          the same function as c but using more or less coefficients than c.

          If n < len(c) the representation is compressed by throwing away
          coefficients, which may result in a loss of accuracy.

          If n > len(c) the coefficients are padded with zeros.
        """

        # Store the number of values the input function has:
        m = len(c)
        coefficients = np.copy(c)

        # n_diff is the number of new values needed (negative if compressing).
        n_diff = n - m

        # Trivial case
        if n_diff == 0:
            # Nothing to do here:
            return coefficients

        if n_diff > 0:
            # append extra zeros
            # This handles the case when coefficients are complex
            out = np.zeros((n,), dtype=coefficients.dtype)
            out[:m] = coefficients

        if n_diff < 0:
            # chop extra coefficients
            m = max([n, 0])
            out = coefficients[:m]

        return out

    def sample_test(self, op, values, v_scale=0.0, h_scale=1.0):
        """Test evaluation of input lambda op against a function approximation.
        evaluates both the function op and its approximation at one or more points
        within [-1, 1]. If the error is sufficiently small (relative to
        v_scale and h_scale) the test passes and returns True, otherwise False.
        """

        # Set a tolerance:
        tol = np.sqrt(np.max(np.array([np.spacing(1), self.tolerance])))

        # Scale tol by the vertical and horizontal scale
        v_scale_of_f = np.max(np.abs(values))
        tol = tol * np.max(np.array([h_scale * v_scale_of_f, v_scale]))

        # choose points to evaluate
        eval_pts = np.array([-0.3002048208008910, 0.0202190191048393])

        # Evaluate the Function:
        v_fun = self(eval_pts)

        # Evaluate the op:
        v_op = op(eval_pts)

        # If the function evaluation differs from the op evaluation, test has failed:
        # error relative to v_scale (tol is a function of v_scale):
        abs_err = np.abs(v_op - v_fun)
        return np.max(abs_err) <= tol

    def simplify(self, tol=None):
        """Remove small trailing Chebyshev coefficients of an approximation.
         simplify attempts to obtain a 'simplified' version g of a resolved
         function f such that len(g) <= len(f) but ||g - f|| is small in
         a relative sense. The algorithm uses the standard_chop() routine.

         If f is not resolved, f is returned unchanged.

         g = f.simplify(tol) does the same as above but uses tol instead of default tol.
        """

        coefficients = np.copy(self.coefficients)
        result = self.copy()

        # Deal with empty case.
        if len(coefficients) == 0:
            return result

        # Do nothing to an unresolved approximation:
        if not self.resolved:
            return result

        # For full reference, read the paper by J. Aurentz and L. N. Trefethen:
        n_old = len(coefficients)
        N = int(np.max(np.r_[self.minimum_samples, np.round(n_old * 1.25 + 5)]))

        coefficients = self.prolong_coefficients(self.coefficients, N)

        coefficients = self.values_to_coefficients(self.coefficients_to_values(coefficients))

        # Use the default tolerance if none was supplied.
        if tol is None:
            tol = self.tolerance

        cutoff = self.standard_chop(coefficients, tol)

        # Take the minimum of cutoff and len(f). This is necessary when padding was required.
        cutoff = np.min(np.r_[cutoff, n_old])

        # Chop coefficients using the cutoff parameter:
        result.coefficients = coefficients[:cutoff]
        return result

    def standard_chop(self, coefficients, tol=None):
        """Rule for chopping a Chebyshev series

        :param coefficients: input coefficients
        :param tol: A number in (0,1) representing a target relative accuracy.
                typically set to machine precision sometimes multiplied
                by a factor such as v_scale etc
        :return: A positive integer called "cutoff"
                If cutoff == len(coefficients), then the series is not resolved,
                a satisfactory chopping point has not been found.
                If cutoff < length(coefficients), then the series is resolved  and
                cutoff represents the last index of coefficients that should be retained.

        See J. L. Aurentz and L. N. Trefethen, "Chopping a Chebyshev series",
        http://arxiv.org/abs/1512.01803, December 2015.
        """

        # Set default if fewer than 2 inputs are supplied: 
        # [TODO]: How to set tolerance to some default:
        if tol is None:
            tol = self.tolerance

        # Check magnitude of tol:
        if tol >= 1:
            cutoff = 1
            return cutoff

        # Make sure input coefficients have length at least the min sample length:
        n = len(coefficients)
        cutoff = n
        if n < self.minimum_samples:
            return cutoff
          
        # Step 1: Convert coefficients to a new monotonically non-increasing
        # vector envelope normalized to begin with the value 1.

        b = np.abs(coefficients)
        m = b[-1] * np.ones((n,))
        for j in np.arange(n-2, -1, -1):
            m[j] = np.max(np.array([b[j], m[j+1]]))

        if m[0] == 0.0:
            cutoff = 1
            return cutoff

        envelope = m / m[0]

        # Step 2: Scan the envelope for a plateau-point

        for j in range(2, n + 1):
            j2 = int(np.round(1.25 * j + 5))
            if j2 > n:
                # there is no plateau: exit
                return cutoff

            e1 = envelope[j-1]
            e2 = envelope[j2-1]
            if e1 == 0.0:
                plateau_point = j - 1
                break
            elif (e2 / e1) > (3.0 * (1.0 - np.log(e1) / np.log(tol))):
                # a plateau has been found: go to Step 3
                plateau_point = j - 1
                break

        # Step 3: fix cutoff at a point where envelope, plus a linear function
        # included to bias the result towards the left end, is minimal.
        # For fuller explanation, see the original paper

        if envelope[plateau_point - 1] == 0.0:
            cutoff = plateau_point
            return cutoff
        else:
            j3 = np.count_nonzero(envelope >= tol**(7.0 / 6.0))
            if j3 < j2:
                j2 = j3 + 1
                envelope[j2-1] = tol**(7.0 / 6.0)

            cc = np.log10(envelope[:j2])
            cc = cc + np.linspace(0, (-1.0/3.0)*np.log10(tol), j2)
            d = np.argmin(cc)
            cutoff = np.max(np.array([d, 1]))
            return cutoff

    def standard_check(self, values=None, v_scale=0.0, h_scale=1.0, tol=None):
        """Attempt to trim Chebyshev coefficients in an approximation.
        """

        if tol is None:
            tol = self.tolerance

        # Grab the coefficients
        coeffs = np.copy(self.coefficients)
        n = len(coeffs)

        # Check for NaNs and exit if any are found.
        assert np.all(~np.isnan(coeffs)), f'Function:standard_check: nan encountered in coeffs: {coeffs}'

        # Compute function values of f if none were given.
        if values is None:
            values = self.coefficients_to_values(coeffs)

        v_scale_of_fun = np.max(np.abs(values))

        # Avoid divide by zero if all values are zero
        if v_scale_of_fun == 0.0:
            v_scale_of_fun = 1.0

        tol = tol * np.max(np.array([h_scale, v_scale / v_scale_of_fun]))

        # Chop the coefficients:
        cutoff = self.standard_chop(coeffs, tol)

        # Check for resolution.
        resolved = (cutoff < n)
        
        return resolved, cutoff

    def refine_by_resampling(self, op, values):
        """Default refinement function for re-sampling scheme."""

        if (values is None) or (len(values) == 0):
            # Choose initial n based upon min_samples:
            n = int(2.0 ** np.ceil(np.log2(self.minimum_samples - 1)) + 1)
        else:
            # Approximate powers of sqrt(2):
            power = np.log2(len(values) - 1)
            if (power == np.floor(power)) and (power > 5):
                n = int(np.round(2.0**(np.floor(power) + 0.5)) + 1)
                n = n - (n % 2) + 1
            else:
                n = int(2.0**(np.floor(power) + 1) + 1)
        
        # n is too large:
        if n > self.maximum_samples:
            # Don't give up if we haven't sampled at least once.
            if len(values) == 0:
                n = self.maximum_samples
                give_up = False
            else:
                give_up = True
                return values, give_up
        else:
            give_up = False
       
        # 2nd-kind Chebyshev grid:
        x = chebyshev_points(n)
        values = op(x)
        
        return values, give_up

    def refine_by_nesting(self, op, values):
        """Default refinement function for nested sampling."""

        if (values is None) or (len(values) == 0):
            # On the first call, there are no values
            # so call refine_by_sampling.
            values, give_up = self.refine_by_resampling(op, values)
            return values, give_up
        else:
            # Compute new n by doubling the existing n
            n = 2 * len(values) - 1
            
            # n is too large and we could not resolve:
            if n > self.maximum_samples:
                give_up = True
                return values, give_up
            else:
                give_up = False
            
            # 2nd-kind Chebyshev grid:
            x = chebyshev_points(n)
            # Take every 2nd entry:
            x = x[1:-1:2]

            if np.all(np.isreal(values)):
                # are we real or not?
                new_values = np.zeros((n,))
            else:
                # not real, so complexify:
                new_values = 0j + np.zeros((n,))

            # Insert the old values:
            new_values[:n:2] = values
            # Compute and insert new values:
            new_values[1::2] = op(x)
            return new_values, give_up


def apply_function_to_all_pieces(func):
    """Decorator which:
    0. Assumes that func creates an object with the same number of "pieces" as the input
    1. creates a copy of the input to func,
    2. uses the name of func and extracts a matching func from FUN
    3. Applies the FUN level func to the function_object
    4. retrun the overall Function object

    :param func:
    :return: deocrated function
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = self.copy()
        for k in range(len(result.pieces)):
            # Get the function of the same name from the function object
            f = (result.pieces[k]).__getattribute__(func.__name__)
            # Apply it:
            result.pieces[k] = f(*args, **kwargs)
        return result
    return wrapper


class Function:
    """Class for approximating functions on arbitrary (finite) intervals."""

    @property
    def length(self) -> int:
        return max([len(f) for f in self.pieces])

    @property
    def points(self) -> np.array:
        n = len(self.pieces)
        assert len(self.domain) == n + 1
        points = []
        for k in range(n):
            a, b = self.domain[k], self.domain[k + 1]
            points.append(self.map_onto_ab((self.pieces[k]).points, a, b))
        return (np.r_[points]).flatten()

    @property
    def values(self) -> np.array:
        values = []
        for k in range(len(self.pieces)):
            values.append((self.pieces[k]).values)
        return (np.r_[values]).flatten()

    @property
    def coefficients(self) -> np.array:
        return np.array([f.coefficients for f in self.pieces])

    @property
    def resolved(self) -> bool:
        return bool(np.prod([f.resolved for f in self.pieces]))

    @property
    def domain(self):
        return self.__domain

    @domain.setter
    def domain(self, dom):
        assert dom[0] < dom[-1]
        assert np.all(np.diff(dom) > 0)
        self.__domain = dom

    @property
    def npieces(self) -> int:
        return self.__npieces

    @npieces.setter
    def npieces(self, n: int):
        assert n >= 0
        self.__npieces = n

    @staticmethod
    def domain_equal(self, other) -> bool:
        domain_1 = self.domain.copy()
        domain_2 = other.domain.copy()
        return np.array_equal(domain_1, domain_2)

    def __init__(self, fun=None, *args, **kwargs):
        # Extract the domain from kwargs and remove it
        self.domain = 1.0 * np.array(kwargs.pop('domain', np.array([-1.0, 1.0])))
        self.npieces = len(self.domain) - 1
        self.pieces = self.npieces * [Fun()]

        for i in range(self.npieces):
            # Update the lambda passed in to map onto [-1, 1]
            a, b = self.domain[i], self.domain[i + 1]
            if fun is not None:
                kwargs['fun'] = lambda x: fun(self.map_onto_ab(x, a, b))

            # Construct the ith piece:
            self.pieces[i] = Fun(*args, **kwargs)

    def copy(self):
        """Deep copy of a function
        :return: an object of type function
        """
        return copy.deepcopy(self)

    def map_onto_ab(self, x, a=None, b=None):
        """values of x in [-1, 1] are linearly mapped on [a, b]"""
        if a is None or b is None:
            a, b = self.domain[0], self.domain[-1]
        return b * (x + 1.0) / 2.0 + a * (1.0 - x) / 2.0

    def map_onto_minus_1_plus_1(self, x, a=None, b=None):
        """values of x in [a, b] are linearly mapped on [-1, 1]"""
        if a is None or b is None:
            a, b = self.domain[0], self.domain[-1]
        return (x - a) / (b - a) - (b - x) / (b - a)

    #############################################################
    ######## section: miscellaenous oeverlaods        ###########
    #############################################################
    # def __getattr__(self, item):
    #     return self.function_object.__getattribute__(item)

    def __len__(self):
        return self.length

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.npieces == 1:
            s = f'Function with {self.npieces} smooth piece with length {self.length} on {self.domain}\n'
        else:
            lengths = [len(f) for f in self.pieces]
            s = f'Function with {self.npieces} smooth pieces with lengths {lengths} on {self.domain}\n'
        return s

    # TODO: should we allow *args and **kwargs in this and Fun __call__ ?
    def __call__(self, x):
        # Call the function object
        if not isinstance(x, types.LambdaType):
            return self[x]
        else:
            assert False, f'__call__, not implemented for lambdas yet'
            # When x is a lambda the result is a
            # composition. TODO: this has not been
            # tested, probably need to fix the lambda for
            # mapping the domain correctly
            # result = self.copy()
            # return result

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        return self.evaluate(x)

    ############################################################
    ####### section: Boolean Operators for Functions ###########
    ############################################################
    def __eq__(self, other):
        out = False
        if len(self.pieces) == len(other.pieces):
            out = np.all([(self.pieces[k]).__eq__(other.pieces[k]) for k in range(len(self.pieces))])
        return out

    def any(self):
        return np.any([f.any() for f in self.pieces])

    def all(self):
        return np.all([f.all() for f in self.pieces])

    def isreal(self):
        """True for a real function"""
        return np.all([f.isreal() for f in self.pieces])

    def iszero(self):
        """True for zero Function objects. """
        # TODO: supply tolerances here?
        return np.all([f.iszero() for f in self.pieces])

    def isequal(self, other):
        out = False
        if len(self.pieces) == len(other.pieces):
            out = np.all([(self.pieces[k]).isequal(other.pieces[k]) for k in range(len(self.pieces))])
        return out

    def isfinite(self):
        """Test if a Function is bounded.
        returns False if f has any unbounded piece
        """
        return np.all([f.isfinite() for f in self.pieces])

    def isinf(self):
        """Test if a Function is unbounded.
        returns True if f has any unbounded piece
        """
        return np.any([f.isinf() for f in self.pieces])

    def isnan(self):
        """Test if a Function has any NaN piece"""
        return np.any([f.isnan() for f in self.pieces])

    #############################################################
    ######## section: Functions that return numbers #############
    #############################################################
    def domain_match_array(self, x: np.array):
        out = self.npieces * [np.zeros((0,))]
        for k in range(self.npieces):
            a, b = self.domain[k], self.domain[k + 1]
            if k == self.npieces - 1:
                mask = np.bitwise_and(a <= x, x <= b)
            else:
                mask = np.bitwise_and(a <= x, x < b)
            xk = x[mask]
            out[k] = xk
        return out

    def evaluate(self, x):
        x_is_scalar = not isinstance(x, Iterable)
        if x_is_scalar:
            # Cast to an array and then numpy array
            xx = 1.0 * np.array([x])
        else:
            # Cast to numpy array, just to allow calls like func([-1, 0, 1]) etc
            xx = 1.0 * np.array(x)

        # prepare output of the same length:
        out = np.zeros((len(xx),))

        # take care of complex
        if not self.isreal():
            out = 1.0j * out
        xk_list = self.domain_match_array(xx)
        assert len(xk_list) == self.npieces
        j = 0
        for k, xk in enumerate(xk_list):
            a, b = self.domain[k], self.domain[k+1]
            if len(xk) > 0:
                out[j:j + len(xk)] = (self.pieces[k]).evaluate(self.map_onto_minus_1_plus_1(xk, a, b))
            j = j + len(xk)
        if x_is_scalar:
            assert len(out) == 1, f'scalar valued evaluation must return array of length 1, returned {out}'
            out = out[0]
        return out

    def vscale(self):
        """Vertical scale of a function.
        :return: float
        """
        return np.max([f.vscale() for f in self.pieces])

    def hscale(self):
        """Horizontal scale of a function.
        :return: float, length of the interval on which the function lives
        """
        return self.domain[-1] - self.domain[0]

    def minandmax(self):
        # TODO, this needs to be fixed:
        # min and max piecewise
        min_max_piecewise = [f.minandmax() for f in self.pieces]
        # painfully reassamble:
        max_idx = np.argmax([v[0][1] for v in min_max_piecewise])
        max_pos = min_max_piecewise[max_idx][1][1]
        max_val = np.max([v[0][1] for v in min_max_piecewise])

        min_idx = np.argmin([v[0][0] for v in min_max_piecewise])
        min_pos = min_max_piecewise[min_idx][1][0]
        min_val = np.min([v[0][0] for v in min_max_piecewise])

        vals = 1.0 * np.array([min_val, max_val])
        pos = 1.0 * np.array([min_pos, max_pos])
        pos = self.map_onto_ab(pos)
        return vals, pos

    def max(self):
        """Global maximum of a Function on its domain."""

        minmax, pos = self.minandmax()
        return minmax[1]

    def argmax(self):
        """Location of global maximum of a Function on its domain."""

        minmax, pos = self.minandmax()
        return pos[1]

    def min(self):
        """Global minimum of a Function on its domain."""

        minmax, pos = self.minandmax()
        return minmax[0]

    def argmin(self):
        """Location of global minimum of a Function on its domain."""

        minmax, pos = self.minandmax()
        return pos[0]

    def norm(self, p: float = 2.0):
        f = self.__pow__(p)
        return np.power(f.definite_integral(), 1.0 / p)

    def definite_integral(self):
        return 0.5 * np.dot(np.diff(self.domain), np.array([f.definite_integral() for f in self.pieces]))

    def sum(self):
        return self.definite_integral()

    def roots(self, **kwargs):
        tol = 10.0 * np.spacing(self.hscale()) * self.hscale()
        roots_pieces = [self.map_onto_ab(f.roots(**kwargs), self.domain[k], self.domain[k + 1])
                        for k, f in enumerate(self.pieces)]
        roots_all = np.concatenate(roots_pieces)
        out = []
        # Check distance of each root from the points in the domain
        # and return the domain point if the root is very close
        for rk in roots_all:
            distance = np.abs(self.domain - rk)
            min_val = np.min(distance)
            min_idx = np.argmin(distance)
            if min_val < tol:
                out.append(self.domain[min_idx])
            else:
                out.append(rk)

        if len(out) > 0:
            out = np.sort(np.array(out))
        else:
            out = np.zeros((0,))

        return out

    def poly(self):
        return 1.0 * np.array([f.poly() for f in self.pieces])

    def coefficients_to_values(self, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients.copy()
        return 1.0 * np.array([f.coefficients_to_values(coefficients[k]) for (k, f) in enumerate(self.pieces)])

    def values_to_coefficients(self, *args, **kwargs):
        # TODO: what does it mean for a piece to have values -> coeffs etc?
        return 1.0 * np.array([f.values_to_coefficients(*args, **kwargs) for f in self.pieces])
    #############################################################
    ######## section: Functions that return Functions ###########
    #############################################################
    @apply_function_to_all_pieces
    def prolong(self, n: int):
        pass

    def abs(self):
        domain = np.sort(np.unique(np.r_[self.domain, self.roots()]))
        return Function(lambda x: np.abs(self(x)), domain=domain)

    @apply_function_to_all_pieces
    def real(self):
        pass

    @apply_function_to_all_pieces
    def imag(self):
        pass

    @apply_function_to_all_pieces
    def conjugate(self):
        pass

    @apply_function_to_all_pieces
    def conj(self):
        pass

    def cumsum(self):
        return self.integral()

    def integral(self):
        result = self.copy()
        n = len(self.pieces)
        assert len(self.domain) == n + 1
        for k in range(n):
            a, b = self.domain[k], self.domain[k + 1]
            scaling_factor = 0.5 * (b - a)
            # TODO: the __rmul__ doesn't seem to work here from the left, so
            # we have mtulitply scaling factor on the right only
            result.pieces[k] = (result.pieces[k]).integral() * scaling_factor
            if k > 0:
                # Evaluate the previous piece at the right end point and add
                # to the current piece (NOTE: the right end point is 1.0)
                result.pieces[k] = result.pieces[k] + (result.pieces[k-1])(1.0)

        return result

    def diff(self, order: int = 1):
        return self.derivative(order)

    def derivative(self, order: int = 1):
        result = self.copy()
        n = len(self.pieces)
        assert len(self.domain) == n + 1
        for k in range(n):
            a, b = self.domain[k], self.domain[k + 1]
            scaling_factor = (0.5 * (b - a)) ** (-order)
            # TODO: the __rmul__ doesn't seem to work here from the left, so
            # we have mtulitply scaling factor on the right only
            result.pieces[k] = (result.pieces[k]).derivative(order) * scaling_factor
        return result

    #############################################################
    ######## section: plotting for Function class     ###########
    #############################################################
    def plot(self, *args, **kwargs):
        a, b = self.domain[0], self.domain[-1]
        x = np.linspace(a, b, 2001)
        y = self(x)
        if not np.all(np.isreal(y)):
            print('Discarding imaginary values in plot')
        y = y.real
        ax = plt.plot(x, y, *args, **kwargs)
        plt.grid(True)
        plt.show()
        return ax

    def plot_coefficients(self, *args, **kwargs):
        return [f.plot_coefficients(*args, **kwargs) for f in self.pieces]

    #############################################################
    ######## section: Overloads that return Functions ###########
    #############################################################
    def __pos__(self):
        # TODO: to copy or not ot copy here? self.copy()
        return self.copy()

    def __pow__(self, a):
        result = self.copy()
        result.pieces = [f.__pow__(a) for f in result.pieces]
        return result

    def __neg__(self):
        result = self.copy()
        result.pieces = [f * (-1.0) for f in result.pieces]
        return result

    def __add__(self, other):
        if not isinstance(other, Function):
            return self.__radd__(other)

        assert Function.domain_equal(self, other)
        result = self.copy()
        result.pieces = [f + g for (f, g) in zip(self.pieces, other.pieces)]
        return result

    def __radd__(self, other):
        result = self.copy()
        result.pieces = [f + other for f in result.pieces]

        return result

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return self.__radd__(-1.0 * other)

    def __mul__(self, other):
        if not isinstance(other, Function):
            return self.__rmul__(other)

        assert Function.domain_equal(self, other)
        result = self.copy()
        result.pieces = [f * g for (f, g) in zip(self.pieces, other.pieces)]
        return result

    def __rmul__(self, other):
        result = self.copy()
        result.pieces = [f * other for f in result.pieces]
        return result

