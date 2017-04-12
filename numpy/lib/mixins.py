"""
Mixin classes for writing custom array types that don't inherit from ndarray.
"""
from __future__ import division, absolute_import, print_function

import operator

from numpy.core import umath as um

__all__ = ['NDArrayOperatorsMixin']


def _binary_method(ufunc):
    def func(self, other):
        try:
            if other.__array_ufunc__ is None:
                return NotImplemented
        except AttributeError:
            pass
        return self.__array_ufunc__(ufunc, '__call__', self, other)
    return func


def _reflected_binary_method(ufunc):
    def func(self, other):
        try:
            if other.__array_ufunc__ is None:
                return NotImplemented
        except AttributeError:
            pass
        return self.__array_ufunc__(ufunc, '__call__', other, self)
    return func


def _inplace_binary_method(ufunc):
    def func(self, other):
        result = self.__array_ufunc__(
            ufunc, '__call__', self, other, out=(self,))
        if result is NotImplemented:
            raise TypeError('unsupported operand types for in-place '
                            'arithmetic: %s and %s'
                            % (type(self).__name__, type(other).__name__))
        return result
    return func


def _numeric_methods(ufunc):
    return (_binary_method(ufunc),
            _reflected_binary_method(ufunc),
            _inplace_binary_method(ufunc))


def _unary_method(ufunc):
    def func(self):
        return self.__array_ufunc__(ufunc, '__call__', self)
    return func


class NDArrayOperatorsMixin(object):
    """Implements all special methods using __array_ufunc__.

    Caveats:
    (1) The rarely used `divmod` (``__divmod__``) and unary `+` (``__pos__``)
        operators do not have corresponding ufuncs. Hence this mixin passes the
        builtin functions ``divmod`` and ``operator.pos`` to
        ``__array_ufunc__`` instead of true ufuncs. If you inherit from this
        mixin, your implementation needs to be able to handle this or you
        should override these methods.
    (2) This mixin doesn't define ``__matmul__``, ``__rmatmul__`` and
        ``__imatmul__``, because ``np.matmul`` is not a ufunc, and hence does
        not call ``__array_ufunc__``.
    """

    # comparisons don't have reflected and in-place versions
    __lt__ = _binary_method(um.less)
    __le__ = _binary_method(um.less_equal)
    __eq__ = _binary_method(um.equal)
    __ne__ = _binary_method(um.not_equal)
    __gt__ = _binary_method(um.greater)
    __ge__ = _binary_method(um.greater_equal)

    # numeric methods
    __add__, __radd__, __iadd__ = _numeric_methods(um.add)
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract)
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply)
    __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide)  # Python 2 only
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(um.true_divide)
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
        um.floor_divide)
    __mod__, __rmod__, __imod__ = _numeric_methods(um.mod)
    __divmod__, __rdivmod__, __idivmod__ = _numeric_methods(divmod)
    # TODO: handle the optional third argument for __pow__?
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power)
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(um.left_shift)
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(um.right_shift)
    __and__, __rand__, __iand__ = _numeric_methods(um.logical_and)
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.logical_xor)
    __or__, __ror__, __ior__ = _numeric_methods(um.logical_or)

    # unary methods
    __neg__ = _unary_method(um.negative)
    __pos__ = _unary_method(operator.pos)
    __abs__ = _unary_method(um.absolute)
    __invert__ = _unary_method(um.invert)
