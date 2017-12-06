import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

class Arctan2(function.Function):

    @property
    def label(self):
        return 'arctan2'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(in_types[0].dtype.kind == 'f')
        type_check.expect(in_types[1].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.arctan2(x[0], x[1])),

    def backward_cpu(self, x, gy):
        x1, x2 = x
        sqnorm = x1 ** 2 + x2 ** 2
        gx1 = utils.force_array(x2 / sqnorm * gy[0])
        gx2 = utils.force_array(-x1 / sqnorm * gy[0])
        return gx1, gx2

    def backward_gpu(self, x, gy):
        gx1, gx2 = cuda.elementwise(
            'T x1, T x2, T gy',
            'T gx1, T gx2',
            ('T sqnorm = x1 * x1 + x2 * x2;'
             'gx1 = x2 / sqnorm * gy;'
             'gx2 = -x1 / sqnorm * gy;'),
            'arctan2_bwd'
        )(x[0], x[1], gy[0])
        return gx1, gx2


def arctan2(x1, x2):
    """Elementwise arctangent function with two arguments.
    Args:
        x1 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Y-coordinates.
        x2 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            X-coordinates.
    Returns:
        ~chainer.Variable: Angles in radians, in the range [-pi, pi].
    """
    return Arctan2()(x1, x2)