from __future__ import division, absolute_import, print_function

import numbers

import numpy as np
from numpy.testing import TestCase, run_module_suite, assert_, assert_equal


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
    """

    def __init__(self, value):
        self.value = np.asarray(value)

    # __array_priority__ = 1000  # for legacy reasons

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
        if isinstance(result, tuple):
            return tuple(type(self)(x) for x in result)
        else:
            return type(self)(result)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.value)


class TestNDArrayOperatorsMixin(TestCase):

    def test_array_like(self):

        x = ArrayLike(1)
        assert_(isinstance(x + 1, ArrayLike))
        assert_equal((x + 1).value, 2)

    def test_opt_out(self):

        class OptOut(object):
            """Object that opts out of __array_ufunc__."""
            __array_ufunc__ = None

            def __add__(self, other):
                return self

            def __radd__(self, other):
                return self

        x = ArrayLike(1)
        o = OptOut()
        assert_(isinstance(x + o, OptOut))
        assert_(isinstance(o + x, OptOut))

    def test_subclass(self):

        class SubArrayLike(ArrayLike):
            """Should take precedence over ArrayLike."""

        # TODO

    def test_ndarray(self):
        pass

    def test_ndarray_subclass(self):
        pass

    def test_out(self):
        pass

    def test_all_operators(self):
        pass


if __name__ == "__main__":
    run_module_suite()
