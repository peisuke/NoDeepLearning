import chainer
from chainer import function
from chainer.utils import type_check

from chainer import cuda
import numpy as np

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class Rotation3DFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 == n_in)
        x_type, r_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            r_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            r_type.ndim == 1,
            x_type.shape[1] == 3,
            r_type.shape[0] == 3,
        )
      
    def rodrigues(self, r, xp):
        def S(n):
            Sn = xp.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
            return Sn
        
        theta = xp.linalg.norm(r)
        
        if theta > 1e-16:
            n = r / theta
            Sn = S(n)
            R = xp.eye(3) + \
                xp.sin(theta) * Sn + \
                (1 - xp.cos(theta)) * xp.dot(Sn, Sn)
        else:
            Sr = S(r)
            theta2 = theta**2
            R = xp.eye(3) + \
                (1- theta2/6.) * Sr + \
                (.5 - theta2/24.) * xp.dot(Sr, Sr)
        
        return R.astype(r.dtype)
        
    def forward(self, inputs):
        x = inputs[0]
        r = inputs[1]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(r): {0}, type(x): {1}'
                             .format(type(r), type(x)))

        xp = cuda.get_array_module(*inputs)
        rmat = self.rodrigues(r, xp)
        y = x.dot(rmat.T).astype(x.dtype, copy=False)
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        r =  inputs[1]
        gy = grad_outputs[0]
        
        xp = cuda.get_array_module(*inputs)
        rmat = self.rodrigues(r, xp)

        gx = gy.dot(rmat).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        
        dR = xp.asarray([
            [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
        ]).astype(r.dtype)

        dRR = np.tensordot(dR, rmat, axes=((2), (0)))        
        dRRx = np.tensordot(dRR, x, axes=((2,),(1,))).transpose((2,0,1))
        
        gr = np.tensordot(gy, dRRx, ((0,1), (0,2)))
                
        return gx, gr
    
def rotation3d(x, r):
    return Rotation3DFunction()(x, r)