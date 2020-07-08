import unittest
import numpy as np
from function import Function
from datetime import datetime
from scipy import special


class TestFunction(unittest.TestCase):
    def test_construction_fixed_length(self):
        """Test construction when length is given
        :return:
        """
        def fun(x):
            return x

        f = Function(fun, length=0)
        self.assertEqual(len(f), 0)

        f = Function(lambda x: x, length=2)
        self.assertEqual(len(f), 2)

        f = Function(fun, length=201)
        self.assertEqual(len(f), 201)
        xx = np.linspace(-1, 1, 201 + np.random.randint(0, 10))
        self.assertTrue(np.allclose(f(xx), xx, atol=1e-15))
        self.assertTrue(np.allclose(f.points, f.values, atol=1e-15))

    def test_construction_adaptive(self):
        """Test the construction process
        """

        f = Function()
        self.assertEqual(len(f), 0)
        self.assertTrue(f is not None)
        self.assertTrue(isinstance(f, Function))

        # Set the tolerance:
        tol = 100 * np.spacing(1)

        # Test on a scalar-valued function:
        f = lambda t: np.sin(t)
        g = Function(f)
        x = g.points
        values = g.coefficients_to_values()
        self.assertTrue(np.linalg.norm(f(x) - values, np.inf) < tol)
        self.assertTrue(np.abs(g.vscale() - np.sin(1.0)) < tol)
        self.assertTrue(g.resolved)

        coeffs = np.array([0.0, 1.0])
        result = Function(coefficients=coeffs)
        self.assertTrue(np.all(result.coefficients == coeffs))

        result = Function(fun=lambda x: x)
        self.assertTrue(np.all(result.coefficients == np.array([0.0, 1.0])))

        xx = np.linspace(-1, 1, 201 + np.random.randint(0, 10))
        f_true = lambda x: np.exp(-10 * x ** 2)
        f = Function(fun=f_true)
        self.assertTrue(np.linalg.norm(f(xx) - f_true(xx), np.inf) < 100 * 1e-16)

        xx = np.linspace(-1, 1, 201 + np.random.randint(0, 10))
        g_true = lambda x: np.sin(4 * np.pi * x)
        g = Function(g_true)
        self.assertTrue(np.linalg.norm(g(xx) - g_true(xx), np.inf) < 100 * 1e-16)

    def test_addition(self):
        """Test that it can sum a list of fractions
        """
        f = Function(lambda x: x)
        g = Function(lambda x: -x)
        h = f + g
        self.assertEqual(h.iszero(), True)
        f = Function(lambda x: np.exp(-10*x**2))
        g = Function(lambda x: np.sin(2.7181828*np.pi*x))
        h = f + g
        xx = np.linspace(-1, 1, 201 + np.random.randint(0, 10))
        error = f(xx) + g(xx) - h(xx)
        self.assertTrue(np.linalg.norm(error, np.inf) < 100 * 1e-16)

    def test_abs(self):
        x = -1 + 2.0 * np.random.rand(100)
        # Test a positive function:

        F = lambda x: np.sin(x) + 2.0
        f = Function(lambda x: F(x))
        h = f.abs()
        self.assertTrue(np.linalg.norm(h(x) - f(x), np.inf) < 10 * np.spacing(1))

        # Test a negative function:
        f2 = Function(lambda x: -F(x))
        h = f2.abs()
        self.assertTrue(np.linalg.norm(h(x) + f2(x), np.inf) < 10 * np.spacing(1))

        # Test a complex-valued function:
        F = lambda x: np.exp(1.0j * np.pi * x)
        f = Function(lambda x: F(x))
        h = f.abs()
        self.assertTrue(np.linalg.norm(h(x) - 1.0, np.inf) < 1e2 * np.spacing(1))

    def test_add(self):
        def test_add_function_to_function(f, f_op, g, g_op, x):
            """Test the addition of two function f and g, specified by f_op and g_op

            :param f:
            :param f_op:
            :param g:
            :param g_op:
            :param x:
            :return:
            """
            h1 = f + g
            h2 = g + f
            result_1 = (h1 == h2)
            h_exact = lambda x: f_op(x) + g_op(x)
            tol = 1e4 * h1.vscale() * np.spacing(1)
            result_2 = (np.linalg.norm(h1(x) - h_exact(x), np.inf) <= tol)

            return result_1 and result_2

        # Generate a few random points to use as test values.
        np.random.seed(6178)
        x = -1 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = -0.194751428283640 + 0.079814485412665j

        # Check operation in the face of empty arguments.

        f = Function()
        g = Function(lambda x: x)
        self.assertEqual(len(f + f), 0)
        self.assertEqual(len(f + g), 0)
        self.assertEqual(len(g + f), 0)

        # Check addition of two function objects.

        f_op = lambda x: np.zeros(len(x))
        f = Function(f_op)
        self.assertTrue(test_add_function_to_function(f, f_op, f, f_op, x))

        f_op = lambda x: np.exp(x) - 1.0
        f = Function(f_op)

        g_op = lambda x: 1.0 / (1.0 + x ** 2)
        g = Function(g_op)
        self.assertTrue(test_add_function_to_function(f, f_op, g, g_op, x))

        g_op = lambda x: np.cos(1e4 * x)
        g = Function(g_op)
        self.assertTrue(test_add_function_to_function(f, f_op, g, g_op, x))

        g_op = lambda t: np.sinh(t * np.exp(2.0 * np.pi * 1.0j / 6.0))
        g = Function(g_op)
        self.assertTrue(test_add_function_to_function(f, f_op, g, g_op, x))

        # Check that direct construction and PLUS give comparable results.
        tol = 10 * np.spacing(1)
        f = Function(lambda x: x)
        g = Function(lambda x: np.cos(x) - 1.0)
        h1 = f + g
        h2 = Function(lambda x: x + np.cos(x) - 1.0)

        # TODO: Improve the constructor so that the following passes:
        self.assertTrue(np.linalg.norm(h1.coefficients - h2.coefficients, np.inf) < tol)

        # Check that adding a resolved fun to an unresolved fun gives an unresolved
        # result.

        # f = Function(lambda x: np.cos(x+1))    # resolved
        # g = Function(lambda x: np.sqrt(x+1))   # Unresolved
        # h = f + g  # Add unresolved to resolved.
        # self.assertTrue (not g.resolved) and (not h.resolved)
        # h = g + f  # Add resolved to unresolved.
        # self.assertTrue = (not g.resolved) and (not h.resolved)

    def test_radd(self):
        # Generate a few random points to use as test values.
        np.random.seed(6178)
        x = -1 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = -0.184752428910640 + 0.079812805462665j

        # Check addition with scalars.

        f_op = lambda x: np.sin(x)
        f = Function(f_op)

        # Test the addition of f, specified by f_op, to a scalar using
        # a grid of points in [-1  1] for testing samples.
        g1 = f + alpha
        g2 = alpha + f
        self.assertTrue(g1 == g2)
        g_exact = lambda x: f_op(x) + alpha
        tol = 10 * g1.vscale() * np.spacing(1)
        self.assertTrue(np.linalg.norm(g1(x) - g_exact(x), np.inf) <= tol)

    def test_rsub(self):
        # Generate a few random points to use as test values.
        x = -1.0 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = np.random.randn() + 1.0j * np.random.randn()

        f_op = lambda x: np.sin(x)
        f = Function(f_op)

        # Test the subtraction of f, to and from a scalar using a grid of points
        g1 = f - alpha
        g2 = alpha - f
        self.assertTrue(g1 == g2)
        g_exact = lambda x: f_op(x) - alpha

        # [TODO] can we bring this tolerance down?
        tol = 1.0e2 * g1.vscale() * np.spacing(1)
        self.assertTrue(np.linalg.norm(g1(x) - g_exact(x), np.inf) <= tol)

    def test_sub(self):
        def test_sub_function_and_function(f, f_op, g, g_op, x):
            """Test the subtraction of two objects f and g, specified by f_op and
            g_op, using a grid of points x in [-1  1] for testing samples.

            :param f:
            :param f_op:
            :param g:
            :param g_op:
            :param x:
            :return:
            """
            h1 = f - g
            h2 = g - f
            result_1 = (h1 == (-1.0 * h2))
            h_exact = lambda x: f_op(x) - g_op(x)
            tol = 1e4 * h1.vscale() * np.spacing(1)
            result_2 = (np.linalg.norm(h1(x) - h_exact(x), np.inf) <= tol)
            return result_1 and result_2

        # Generate a few random points to use as test values.
        x = -1.0 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = np.random.randn() + 1.0j * np.random.randn()

        # Check operation in the face of empty arguments.

        f = Function()
        g = Function(lambda x: x)
        self.assertEqual(len(f - f), 0)
        self.assertEqual(len(f - g), 0)
        self.assertEqual(len(g - f), 0)

        # Check subtraction of two function objects.

        f_op = lambda x: np.zeros(len(x))
        f = Function(f_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, f, f_op, x))

        f_op = lambda x: np.exp(x) - 1
        f = Function(f_op)

        g_op = lambda x: 1.0 / (1 + x ** 2)
        g =Function(g_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, g, g_op, x))

        g_op = lambda x: np.cos(1e4 * x)
        g = Function(g_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, g, g_op, x))

        g_op = lambda t: np.sinh(t * np.exp(2.0 * np.pi * 1.0j / 6.0))
        g = Function(g_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, g, g_op, x))

        # Check that direct construction and the binary minus op give comparable results.

        tol = 10.0 * np.spacing(1)
        f = Function(lambda x: x)
        g = Function(lambda x: np.cos(x) - 1)
        h1 = f - g
        h2 = Function(lambda x: x - (np.cos(x) - 1))
        h3 = h1 - h2
        self.assertTrue(np.linalg.norm(h3.coefficients, np.inf) < tol)

        # [TODO]
        # Check that subtracting a resolved function and an unresolved function gives an
        # unresolved result.

        # f = Function(lambda x: np.cos(x+1))    # resolved
        # g = Function(lambda x: np.sqrt(x+1))   # Unresolved
        # h = f - g  # Subtract unresolved from resolved.
        # self.assertTrue( 20) = (not g.resolved) and (not h.resolved)
        # h = g - f  # Subtract resolved from unresolved.
        # self.assertTrue( 21) = (not g.resolved) and (not h.resolved)

    def test_rmul(self):
        """ Test the multiplication of f, specified by f_op, by a scalar alpha
        Generate a few random points to use as test values.
        :return:
        """
        x = -1 + 2.0 * np.random.rand(100)

        # Random numbers to use as arbitrary multiplicative constants.
        alpha = -0.213251928283644 + 0.053493485412265j

        # Check multiplication by scalars.
        f_op = lambda x: np.sin(x)
        f = Function(f_op)
        g1 = f * alpha
        g2 = alpha * f
        self.assertTrue(g1 == g2)
        g_exact = lambda x: f_op(x) * alpha
        self.assertTrue(np.linalg.norm(g1(x) - g_exact(x), np.inf) < 10 * np.max(g1.vscale() * np.spacing(1)))

    def test_mul(self):
        def test_mul_function_by_function(f, f_op, g, g_op, x, check_positive):
            """ Test the multiplication of two function f and g, specified by f_op and
             g_op, using a grid of points x in [-1  1] for testing samples.  If check_positive is
             True, an additional check is performed to ensure that the values of the result
             are all non-negative otherwise, this check is skipped.

            :param f:
            :param f_op:
            :param g:
            :param g_op:
            :param x:
            :param check_positive:
            :return:
            """
            h = f * g
            h_exact = lambda x: f_op(x) * g_op(x)
            tol = 1e4 * np.max(h.vscale() * np.spacing(1))
            result_1 = np.linalg.norm(h(x) - h_exact(x), np.inf) < tol
            result_2 = True
            if check_positive:
                values = h.coefficients_to_values(h.coefficients)
                result_2 = np.all(values >= -tol)
            return result_1 and result_2

        # Generate a few random points to use as test values.
        np.random.seed(6178)
        x = -1 + 2.0 * np.random.rand(100)

        # Random numbers to use as arbitrary multiplicative constants.
        alpha = -0.194758928283640 + 0.075474485412665j

        # Check operation in the face of empty arguments.

        f = Function()
        g = Function(lambda x: x)
        self.assertEqual(len(f * f), 0)
        self.assertEqual(len(f * g), 0)
        self.assertEqual(len(g * f), 0)

        # Check multiplication by constant functions.

        f_op = lambda x: np.sin(x)
        f = Function(f_op)
        g_op = lambda x: np.zeros(len(x)) + alpha
        g = Function(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))

        # Spot-check multiplication of two function objects for a few test
        # functions.

        f_op = lambda x: np.ones(len(x))
        f = Function(f_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, f, f_op, x, False))

        f_op = lambda x: np.exp(x) - 1.0
        f = Function(f_op)

        g_op = lambda x: 1.0 / (1.0 + x ** 2)
        g = Function(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))

        # If f and g are real then so must be f * g
        h = f * g
        self.assertTrue(h.isreal())

        g_op = lambda x: np.cos(1.0e4 * x)
        g = Function(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))

        g_op = lambda t: np.sinh(t * np.exp(2.0 * np.pi * 1.0j / 6.0))
        g = Function(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))

        # Check specially handled cases, including some in which an adjustment for
        # positivity is performed.

        f_op = lambda t: np.sinh(t * np.exp(2.0 * np.pi * 1.0j / 6.0))
        f = Function(f_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, f, f_op, x, False))

        g_op = lambda t: np.conjugate(np.sinh(t * np.exp(2.0 * np.pi * 1.0j / 6.0)))
        g = f.conj()
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, True))

        f_op = lambda x: np.exp(x) - 1.0
        f = Function(f_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, f, f_op, x, True))

        # Check that multiplication and direct construction give similar results.

        tol = 50 * np.spacing(1)
        g_op = lambda x: 1.0 / (1.0 + x ** 2)
        g = Function(g_op)
        h1 = f * g
        h2 = Function(lambda x: f_op(x) * g_op(x))
        h2 = h2.prolong(len(h1))
        self.assertTrue(np.linalg.norm(h1.coefficients - h2.coefficients, np.inf) < tol)

        # Check that multiplying a function by an unresolved function gives an unresolved result.

        # [TODO] implement resolved vs unresolved etc :)

        # f = Function(lambda x: cos(x+1))    # resolved
        # g = Function(lambda x: sqrt(x+1))   # Unresolved
        # h = f.*g  # Multiply resolved by unresolved
        # assert here ->   = (not g.resolved) and (not h.resolved)
        # h = g.*f  # Multiply unresolved by resolved.
        # assert here -> = (not g.resolved) and (not h.resolved)

    def test_roots(self):
        func = lambda x: (x + 1) * 50
        f = Function(lambda x: special.j0(func(x)))
        r = func(f.roots())
        exact = np.array([
            2.40482555769577276862163, 5.52007811028631064959660,
            8.65372791291101221695437, 11.7915344390142816137431,
            14.9309177084877859477626, 18.0710639679109225431479,
            21.2116366298792589590784, 24.3524715307493027370579,
            27.4934791320402547958773, 30.6346064684319751175496,
            33.7758202135735686842385, 36.9170983536640439797695,
            40.0584257646282392947993, 43.1997917131767303575241,
            46.3411883716618140186858, 49.4826098973978171736028,
            52.6240518411149960292513, 55.7655107550199793116835,
            58.9069839260809421328344, 62.0484691902271698828525,
            65.1899648002068604406360, 68.3314693298567982709923,
            71.4729816035937328250631, 74.6145006437018378838205,
            77.7560256303880550377394, 80.8975558711376278637723,
            84.0390907769381901578795, 87.1806298436411536512617,
            90.3221726372104800557177, 93.4637187819447741711905,
            96.6052679509962687781216, 99.7468198586805964702799])

        self.assertTrue(np.linalg.norm(r - exact, np.inf) < 1.0e1 * len(f) * np.spacing(1))

        k = 500
        f = Function(lambda x: np.sin(np.pi * k * x))
        r = f.roots()
        self.assertTrue(np.linalg.norm(r - (1.0 * np.r_[-k:k + 1]) / k, np.inf) < 1e1 * len(f) * np.spacing(1))

        # Test a perturbed polynomial:
        f = Function(lambda x: (x - .1) * (x + .9) * x * (x - .9) + 1e-14 * x ** 5)
        r = f.roots()
        self.assertEqual(len(r), 4)
        self.assertTrue(np.linalg.norm(f(r), np.inf) < 1e2 * len(f) * np.spacing(1))

        # Test a some simple polynomials:
        f = Function(values=[-1.0, 1.0])
        r = f.roots()
        self.assertTrue(np.all(r == 0))

        # f = testclass.make([1  0  1])
        f = Function(values=[1.0, 0.0, 1.0])
        r = f.roots()
        self.assertEqual(len(r), 2)
        self.assertTrue(np.linalg.norm(r, np.inf) < np.spacing(1))

        # Test some complex roots:
        f = Function(lambda x: 1 + 25 * x ** 2)
        r = f.roots(complex_roots=True)
        self.assertEqual(len(r), 2)
        self.assertTrue(np.linalg.norm(r - np.r_[-1.0j, 1.0j] / 5.0, np.inf) < 10 * np.spacing(1))

        # [TODO] This is failing:
        # f = Function(lambda x: (1 + 25*x**2)*np.exp(x))
        # r = f.roots(complex_roots=True, prune=True)
        # self.assertEqual(len(r), 2)
        # self.assertTrue(np.linalg.norm( r - np.r_[1.0j, -1.0j]/5.0, np.inf) < 10*len(f)*np.spacing(1))

        # [TODO] We get different number of roots
        # f = Function(lambda x: np.sin(100*np.pi*x))
        # r1 = f.roots(complex_roots=True, recurse=False)
        # r2 = f.roots(complex_roots=True)

        # self.assertEqual(len(r1), 201)
        # self.assertEqual(len(r2), 213)

        # Adding test for 'qz' flag:
        f = Function(lambda x: 1e-10 * x ** 3 + x ** 2 - 1e-12)
        r = f.roots(qz=True)
        self.assertFalse(len(r) == 0)
        self.assertTrue(np.linalg.norm(f[r], np.inf) < 10 * np.spacing(1))

        # Add a rootfinding test for low degree non-even functions:
        f = Function(lambda x: (x - .5) * (x - 1.0 / 3.0))
        r = f.roots(qz=True)
        self.assertTrue(np.linalg.norm(f[r], np.inf) < np.spacing(1))

    def test_sum(self):
        # Spot-check integrals for a couple of functions.
        f = Function(lambda x: np.exp(x) - 1.0)
        self.assertTrue(np.abs(f.sum() - 0.350402387287603) < 10 * f.vscale() * np.spacing(1))

        f = Function(lambda x: 1. / (1 + x ** 2))
        self.assertTrue(np.abs(f.sum() - np.pi / 2.0) < 10 * f.vscale() * np.spacing(1))

        f = Function(lambda x: np.cos(1e4 * x))
        exact = -6.112287777765043e-05
        self.assertTrue(np.abs(f.sum() - exact) / np.abs(exact) < 1e6 * f.vscale() * np.spacing(1))

        z = np.exp(2 * np.pi * 1.0j / 6.0)
        f = Function(lambda t: np.sinh(t * z))
        self.assertTrue(np.abs(f.sum()) < 10 * f.vscale() * np.spacing(1))

        # Check a few basic properties.
        a = 2.0
        b = -1.0j
        f = Function(lambda x: x * np.sin(x ** 2) - 1)
        df = f.diff()
        g = Function(lambda x: np.exp(-x ** 2))
        dg = g.diff()
        fg = f * g
        gdf = g * df
        fdg = f * dg

        tol_f = 10 * f.vscale() * np.spacing(1)
        tol_g = 10 * f.vscale() * np.spacing(1)
        tol_df = 10 * df.vscale() * np.spacing(1)
        tol_dg = 10 * dg.vscale() * np.spacing(1)
        tol_fg = 10 * fg.vscale() * np.spacing(1)
        tol_fdg = 10 * fdg.vscale() * np.spacing(1)
        tol_gdf = 10 * gdf.vscale() * np.spacing(1)

        # Linearity.
        self.assertTrue(np.abs((a * f + b * g).sum() - (a * f.sum() + b * g.sum())) < max(tol_f, tol_g))

        # Integration-by-parts.
        self.assertTrue(np.abs(fdg.sum() - (fg(1) - fg(-1) - gdf.sum())) < np.max(np.r_[tol_fdg, tol_gdf, tol_fg]))

        # Fundamental Theorem of Calculus.
        self.assertTrue(np.abs(df.sum() - (f(1) - f(-1))) < np.max(np.r_[tol_df, tol_f]))
        self.assertTrue(np.abs(dg.sum() - (g(1) - g(-1))) < np.max(np.r_[tol_dg, tol_g]))

    def test_cumsum(self):
        # Generate a few random points to use as test values.

        x = 2 * np.random.rand(100) - 1

        # Spot-check antiderivatives for a couple of functions.  We verify that the
        # function antiderivatives match the true ones up to a constant by checking
        # that the standard deviation of the difference between the two on a large
        # random grid is small. We also check that evaluate(f.cumsum(), -1) == 0 each
        # time.

        f = Function(lambda x: np.exp(x) - 1.0)
        F = f.cumsum()
        F_ex = lambda x: np.exp(x) - x
        err = np.std(F[x] - F_ex(x))
        tol = 20 * F.vscale() * np.spacing(1)
        self.assertTrue(err < tol)
        self.assertTrue(np.abs(F[-1]) < tol)

        f = Function(lambda x: 1.0 / (1.0 + x ** 2))
        F = f.cumsum()
        F_ex = lambda x: np.arctan(x)
        err = np.std(F[x] - F_ex(x))
        tol = 10 * F.vscale() * np.spacing(1)
        self.assertTrue(err < tol)
        self.assertTrue(np.abs(F[-1]) < tol)

        f = Function(lambda x: np.cos(1.0e4 * x))
        F = f.cumsum()
        F_ex = lambda x: np.sin(1.0e4 * x) / 1.0e4
        err = F[x] - F_ex(x)
        tol = 10.0e4 * F.vscale() * np.spacing(1)
        self.assertTrue((np.std(err) < tol) and (np.abs(F[-1]) < tol))

        z = np.exp(2 * np.pi * 1.0j / 6)
        f = Function(lambda t: np.sinh(t * z))
        F = f.cumsum()
        F_ex = lambda t: np.cosh(t * z) / z
        err = F[x] - F_ex(x)
        tol = 10 * F.vscale() * np.spacing(1)
        self.assertTrue((np.std(err) < tol) and (np.abs(F[-1]) < tol))

        # Check that applying cumsum() and direct construction of the antiderivative
        # give the same results (up to a constant).

        f = Function(lambda x: np.sin(4.0 * x) ** 2)
        F = Function(lambda x: 0.5 * x - 0.0625 * np.sin(8 * x))
        G = f.cumsum()
        err = G - F
        tol = 10 * G.vscale() * np.spacing(1)
        values = err.coefficients_to_values(err.coefficients)
        self.assertTrue((np.std(values) < tol) and (np.abs(G[-1]) < tol))

        # Check that f.diff().cumsum() == f and that f.cumsum().diff() == f up to a
        # constant.

        f = Function(lambda x: x * (x - 1.0) * np.sin(x) + 1.0)
        integral_f = lambda x: (-x ** 2 + x + 2) * np.cos(x) + x + (2 * x - 1) * np.sin(x)
        F = lambda x: integral_f(x) - integral_f(-1)
        g = f.cumsum().diff()
        err = f(x) - g(x)
        tol = 10 * g.vscale() * np.spacing(1)
        self.assertTrue(np.linalg.norm(err, np.inf) < 100 * tol)

        h = f.diff().cumsum()
        err = f(x) - h(x)
        tol = 10 * h.vscale() * np.spacing(1)
        self.assertTrue((np.std(err) < tol) and (np.abs(h[-1]) < tol))

    def test_max(self):
        def spotcheck_max(fun_op, exact_max):
            """Spot-check the results for a given function.
            :param fun_op:
            :param exact_max:
            :return:
            """
            f = Function(fun_op)
            y = f.max()
            x = f.argmax()
            fx = fun_op(x)

            # [TODO]: Try to get this tolerance down:
            result = (np.all(np.abs(y - exact_max) < 1.0e2 * f.vscale() * np.spacing(1))) and (
                np.all(np.abs(fx - exact_max) < 1.0e2 * f.vscale() * np.spacing(1)))

            return result

        # Spot-check the extrema for a few functions.
        self.assertTrue(
            spotcheck_max(lambda x: ((x - 0.2) ** 3 - (x - 0.2) + 1) * 1.0 / np.cos(x - 0.2), 1.884217141925336))
        self.assertTrue(spotcheck_max(lambda x: np.sin(10 * x), 1.0))
        # self.assertTrue(spotcheck_max(lambda x: airy, airy(-1)))
        self.assertTrue(spotcheck_max(lambda x: -1.0 / (1.0 + x ** 2), -0.5))
        self.assertTrue(spotcheck_max(lambda x: (x - 0.25) ** 3 * np.cosh(x), 0.75 ** 3 * np.cosh(1.0)))

        # Test for complex-valued function objects.
        self.assertTrue(spotcheck_max(lambda x: (x - 0.2) * (np.exp(1.0j * (x - 0.2)) + 1.0j * np.sin(x - 0.2)),
                                      -0.434829305372008 + 2.236893806321343j))

    def test_min(self):
        def spotcheck_min(fun_op, exact_min):
            # Spot-check the results for a given function.
            f = Function(fun_op)
            y = f.min()
            x = f.argmin()
            fx = fun_op(x)
            result = ((np.abs(y - exact_min) < 1.0e2 * f.vscale() * np.spacing(1)) and (
                    np.abs(fx - exact_min) < 1.0e2 * f.vscale() * np.spacing(1)))

            return result

        # Spot-check the extrema for a few functions.

        self.assertTrue(
            spotcheck_min(lambda x: -((x - 0.2) ** 3 - (x - 0.2) + 1) * 1.0 / np.cos(x - 0.2), -1.884217141925336))
        self.assertTrue(spotcheck_min(lambda x: -np.sin(10 * x), -1.0))
        # self.assertTrue(spotcheck_min(lambda x:  -airy(x), -airy(-1), pref)
        self.assertTrue(spotcheck_min(lambda x: 1.0 / (1 + x ** 2), 0.5))
        self.assertTrue(spotcheck_min(lambda x: -(x - 0.25) ** 3. * np.cosh(x), -0.75 ** 3 * np.cosh(1.0)))

        # Test for complex-valued function objects.
        self.assertTrue(
            spotcheck_min(lambda x: np.exp(1.0j * x) - 0.5j * np.sin(x) + x, 0.074968381369117 - 0.319744137826069j))

    def test_poly(self):
        f = Function()
        self.assertEqual(len(f.poly()[0]), 0)

        f = Function(lambda x: 1.0 + 0.0 * x)
        self.assertTrue(np.linalg.norm(f.poly()[0] - np.ones(1), np.inf) < f.vscale() * np.spacing(1))
        f = Function(lambda x: 1 + x)
        self.assertTrue(np.linalg.norm(f.poly()[0] - np.ones(2), np.inf) < f.vscale() * np.spacing(1))
        f = Function(lambda x: 1 + x + x**2)
        self.assertTrue(np.linalg.norm(f.poly()[0] - np.ones(3), np.inf) < f.vscale() * np.spacing(1))
        f = Function(lambda x: 1 + x + x**2 + x**3)
        self.assertTrue(np.linalg.norm(f.poly()[0] - np.ones(4), np.inf) < f.vscale() * np.spacing(1))
        f = Function(lambda x: 1 + x + x**2 + x**3 + x**4)
        self.assertTrue(np.linalg.norm(f.poly()[0] - np.ones(5), np.inf) < f.vscale() * np.spacing(1))

def run_all_tests():
    seed = datetime.now().microsecond
    np.random.seed(seed=seed)
    print(f'Using {seed} as the seed value for the test')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    t = TestFunction()
    t.test_sum()
    run_all_tests()

