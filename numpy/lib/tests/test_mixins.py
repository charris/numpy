from __future__ import division, absolute_import, print_function

import numbers
import operator
import sys

import numpy as np
from numpy.testing import (
    TestCase, run_module_suite, assert_, assert_equal, assert_raises)


PY2 = sys.version_info.major < 3


class ArrayLike(np.NDArrayOperatorsMixin):
    """An array-like class that wraps NumPy arrays.

    Example usage:

        >>> x = ArrayLike([1, 2, 3])
        >>> x - 1
        ArrayLike(array([0, 1, 2]))
        >>> 1 - x
        ArrayLike(array([ 0, -1, -2]))
        >>> np.arange(3) - x
        ArrayLike(array([-1, -1, -1]))
        >>> x - np.arange(3)
        ArrayLike(array([1, 1, 1]))

    Note that unlike numpy.ndarray, this type does not allow operations with
    arbitrary, unrecognized types. This ensures that ArrayLike maintains a
    well-defined casting hierarchy, as described in the NEP "A Mechanism for
    Overriding Ufuncs".
    """

    def __init__(self, value):
        self.value = np.asarray(value)

    # We might also consider adding the built-in list type to this list
    _handled_types = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # ArrayLike implements arithmetic and ufuncs by deferring to the
        # wrapped array
        out = kwargs.get('out', ())
        for x in inputs + out:
            # handle _handled_types and superclass instances
            if not (isinstance(x, self._handled_types) or
                    isinstance(self, type(x))):
                return NotImplemented

        inputs = tuple(x.value if isinstance(self, type(x)) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(x.value if isinstance(self, type(x)) else x
                                  for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method != 'at':
            # one return value
            return type(self)(result)
        else:
            # no return value
            return None

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.value)


def _assert_equal_type_and_value(result, expected, err_msg=None):
    assert_equal(type(result), type(expected), err_msg=err_msg)
    assert_equal(result.value, expected.value, err_msg=err_msg)
    assert_equal(getattr(result.value, 'dtype', None),
                 getattr(expected.value, 'dtype', None), err_msg=err_msg)


class TestNDArrayOperatorsMixin(TestCase):

    def test_array_like_add(self):

        def check(result):
            _assert_equal_type_and_value(result, ArrayLike(0))

        check(ArrayLike(0) + 0)
        check(0 + ArrayLike(0))

        check(ArrayLike(0) + np.array(0))
        check(np.array(0) + ArrayLike(0))

        check(ArrayLike(np.array(0)) + 0)
        check(0 + ArrayLike(np.array(0)))

        check(ArrayLike(np.array(0)) + np.array(0))
        check(np.array(0) + ArrayLike(np.array(0)))

    def test_inplace(self):
        array_like = ArrayLike(np.array([0]))
        array_like += 1
        _assert_equal_type_and_value(array_like, ArrayLike(np.array([1])))

        array = np.array([0])
        array += ArrayLike(1)
        _assert_equal_type_and_value(array, ArrayLike(np.array([1])))

    def test_opt_out(self):

        class OptOut(object):
            """Object that opts out of __array_ufunc__."""
            __array_ufunc__ = None

            def __add__(self, other):
                return self

            def __radd__(self, other):
                return self

        array_like = ArrayLike(1)
        opt_out = OptOut()

        # supported operations
        assert_(array_like + opt_out is opt_out)
        assert_(opt_out + array_like is opt_out)

        # not supported
        with assert_raises(TypeError):
            # don't use the Python default, array_like = array_like + opt_out
            array_like += opt_out
        with assert_raises(TypeError):
            array_like - opt_out
        with assert_raises(TypeError):
            opt_out - array_like

    def test_subclass(self):

        class SubArrayLike(ArrayLike):
            """Should take precedence over ArrayLike."""

        x = ArrayLike(0)
        y = SubArrayLike(1)
        _assert_equal_type_and_value(x + y, y)
        _assert_equal_type_and_value(y + x, y)

    def test_unary_methods(self):
        array = np.array([-1, 0, 1, 2])
        array_like = ArrayLike(array)
        for op in [operator.neg,
                   # pos is not yet implemented
                   abs,
                   operator.invert]:
            _assert_equal_type_and_value(op(array_like), ArrayLike(op(array)))

    def test_binary_methods(self):
        array = np.array([-1, 0, 1, 2])
        array_like = ArrayLike(array)
        operators = [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.floordiv,
            # TODO: test div on Python 2, only
            operator.mod,
            # divmod is not yet implemented
            pow,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.xor,
            operator.or_,
        ]
        for op in operators:
            expected = ArrayLike(op(array, 1))
            actual = op(array_like, 1)
            err_msg = 'failed for operator {}'.format(op)
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)

    def test_ufunc_at(self):
        array = ArrayLike(np.array([1, 2, 3, 4]))
        assert_(np.negative.at(array, np.array([0, 1])) is None)
        _assert_equal_type_and_value(array, ArrayLike([-1, -2, 3, 4]))

    def test_ufunc_two_outputs(self):
        def check(result):
            assert_(type(result) is tuple)
            assert_equal(len(result), 2)
            mantissa, exponent = np.frexp(2 ** -3)
            _assert_equal_type_and_value(result[0], ArrayLike(mantissa))
            _assert_equal_type_and_value(result[1], ArrayLike(exponent))

        check(np.frexp(ArrayLike(2 ** -3)))
        check(np.frexp(ArrayLike(np.array(2 ** -3))))


if __name__ == "__main__":
    run_module_suite()
