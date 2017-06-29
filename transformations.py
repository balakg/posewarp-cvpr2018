# coding: utf-8
'''Spatial transformation library.

Description
===========

Create and apply several spatial 2D and 3D transformations including similarity,
bilinear, projective, polynomial and affine transformation. You can determine
the over-, well- and under-determined parameters with the least-squares method.

Create 2D and 3D rotation matrices.

Usage
=====

>>> tform = make_tform('similarity', np.array([[1,1], [2,2]]),
... np.array([[3,4], [10,10]]))
>>> tform.params
array([-3.25,  3.25, -2.75, -3.25])
>>> tform.params_explicit
array([-3.25      , -2.75      ,  4.59619408, -0.78539816])
>>> tform.fwd(np.array([[0, 0], [100,100]]))
array([[  -3.25,   -2.75],
       [ 646.75,   -2.75]])
>>> tform.inv(tform.fwd(np.array([[0, 0], [100,100]])))
array([[   0.,    0.],
       [ 100.,  100.]])

Reference
=========

"Nahbereichsphotogrammetrie - Grundlagen, Methoden und Anwendungen",
    Thomas Luhmann, 2010
'''
import warnings
import numpy as np
import math


TRANSFORMATIONS = [
    'similarity',
    'bilinear',
    'projective',
    'polynomial',
    'affine',
]

def make_tform(ttype, src, dst):
    '''
    Create spatial transformation.
    You can determine the over-, well- and under-determined parameters
    with the least-squares method.
    The following transformation types are supported:

        NAME / TTYPE            DIM     NUM POINTS FOR EXACT SOLUTION
        similarity:              2D      2
        bilinear:               2D      4
        projective:             2D      4
        polynomial (order n):  2D      (n+1)*(n+2)/2
        affine:                 2D      3
        affine:                 3D      4

    Number of source must match number of destination coordinates.

    :param ttype: similarity, bilinear, projective, polynomial, affine
        transformation type
    :param src: :class:`numpy.array`
        Nx2 or Nx3 coordinate matrix of source coordinate system
    :param src: :class:`numpy.array`
        Nx2 or Nx3 coordinate matrix of destination coordinate system

    :returns: :class:`Transformation`
    '''

    ttype = ttype.lower()
    if ttype not in TRANSFORMATIONS:
        raise NotImplemented(
            'Your transformation type %s is not implemented' % ttype)
    params, params_explicit = MFUNCS[ttype](src, dst)
    return Transformation(ttype, params, params_explicit)

def make_similarity(src, dst,flip=False):
    '''
    Determine parameters of 2D similarity transformation in the order:
        a0, a1, b0, b1
    where the transformation is defined as:
        X = a0 + a1*x - b1*y
        Y = b0 + b1*x + a1*y
    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    Explicit parameters are in the order:
        a0, b0, m, alpha [radians]
    where the transformation is defined as:
        X = a0 + m*x*cos(alpha) - m*y*sin(alpha)
        Y = b0 + m*x*sin(alpha) + m*y*cos(alpha)

    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of destination coordinate system

    :returns: params, params_explicit
    '''

    xs = src[:,0]
    ys = src[:,1]
    rows = src.shape[0]
    A = np.zeros((rows*2, 4))
    A[:rows,0] = 1
    A[:rows,1] = xs
    A[:rows,3] = -ys
    A[rows:,2] = 1
    A[rows:,3] = xs
    A[rows:,1] = ys
    
    if(flip):
      A[:rows,3] *= -1.0
      A[rows:,1] *= -1.0

    b = np.zeros((rows*2,))
    b[:rows] = dst[:,0]
    b[rows:] = dst[:,1]
    params = np.linalg.lstsq(A, b)[0]
    '''
    #: determine explicit params
    a0, b0 = params[0], params[2]
    alpha = math.atan2(params[3], params[1])
    m = params[1] / math.cos(alpha)
    params_explicit = np.array([a0, b0, m, alpha])
    '''    

    return params #, params_explicit

def similarity_transform(coords, params, inverse=False):
    '''
    Apply similarity transformation.

    :param coords: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param params: :class:`numpy.array`
        parameters returned by `make_tform`
    :param inverse: bool
        apply inverse transformation, default is False

    :returns: :class:`numpy.array`
        transformed coordinates
    '''

    a0, a1, b0, b1 = params
    x = coords[:,0]
    y = coords[:,1]
    out = np.zeros(coords.shape)
    if inverse:
        out[:,0] = (a1*(x-a0)+b1*(y-b0)) / (a1**2+b1**2)
        out[:,1] = (a1*(y-b0)-b1*(x-a0)) / (a1**2+b1**2)
    else:
        out[:,0] = a0+a1*x-b1*y
        out[:,1] = b0+b1*x+a1*y
    return out

def make_bilinear(src, dst):
    '''
    Determine parameters of 2D bilinear transformation in the order:
        a0, a1, a2, a3, b0, b1, b2, b3
    where the transformation is defined as:
        X = a0 + a1*x + a2*y + a3*x*y
        Y = b0 + b1*x + b2*y + b3*x*y
    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of destination coordinate system

    :returns: params, None
    '''

    xs = src[:,0]
    ys = src[:,1]
    # affine transformation is polynomial transformation of order 1
    rows = src.shape[0]
    A = np.zeros((rows*2, 8))
    A[:rows,0] = 1
    A[:rows,1] = xs
    A[:rows,2] = ys
    A[:rows,3] = xs*ys
    A[rows:,4] = 1
    A[rows:,5] = xs
    A[rows:,6] = ys
    A[:rows,7] = xs*ys
    b = np.zeros((rows*2,))
    b[:rows] = dst[:,0]
    b[rows:] = dst[:,1]
    params = np.linalg.lstsq(A, b)[0]
    return params, None

def bilinear_transform(coords, params, inverse=False):
    '''
    Apply bilinear transformation.

    :param coords: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param params: :class:`numpy.array`
        parameters returned by `make_tform`
    :param inverse: bool
        apply inverse transformation, default is False

    :returns: :class:`numpy.array`
        transformed coordinates
    '''

    a0, a1, a2, a3, b0, b1, b2, b3 = params
    x = coords[:,0]
    y = coords[:,1]
    out = np.zeros(coords.shape)
    if inverse:
        raise NotImplemented('There is no explicit way to do the inverse '
            'transformation. Determine the inverse transformation parameters '
            'and use the fwd transformation instead.')
    else:
        out[:,0] = a0+a1*x+a2*y+a3*x*y
        out[:,1] = b0+b1*x+b2*y+b3*x*y
    return out

def make_projective(src, dst):
    '''
    Determine parameters of 2D projective transformation in the order:
        a0, a1, a2, b0, b1, b2, c0, c1
    where the transformation is defined as:
        X = (a0+a1*x+a2*y) / (1+c0*x+c1*y)
        Y = (b0+b1*x+b2*y) / (1+c0*x+c1*y)
    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of destination coordinate system

    :returns: params, None
    '''

    xs = src[:,0]
    ys = src[:,1]
    rows = src.shape[0]
    A = np.zeros((rows*2, 8))
    A[:rows,0] = 1
    A[:rows,1] = xs
    A[:rows,2] = ys
    A[rows:,3] = 1
    A[rows:,4] = xs
    A[rows:,5] = ys
    A[:rows,6] = - dst[:,0] * xs
    A[:rows,7] = - dst[:,0] * ys
    A[rows:,6] = - dst[:,1] * xs
    A[rows:,7] = - dst[:,1] * ys
    b = np.zeros((rows*2,))
    b[:rows] = dst[:,0]
    b[rows:] = dst[:,1]
    params = np.linalg.lstsq(A, b)[0]
    return params, None

def projective_transform(coords, params, inverse=False):
    '''
    Apply projective transformation.

    :param coords: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param params: :class:`numpy.array`
        parameters returned by `make_tform`
    :param inverse: bool
        apply inverse transformation, default is False

    :returns: :class:`numpy.array`
        transformed coordinates
    '''

    a0, a1, a2, b0, b1, b2, c0, c1 = params
    x = coords[:,0]
    y = coords[:,1]
    out = np.zeros(coords.shape)
    if inverse:
        out[:,0] = (a2*b0-a0*b2+(b2-b0*c1)*x+(a0*c1-a2)*y) \
            / (a1*b2-a2*b1+(b1*c1-b2*c0)*x+(a2*c0-a1*c1)*y)
        out[:,1] = (a0*b1-a1*b0+(b0*c0-b1)*x+(a1-a0*c0)*y) \
            / (a1*b2-a2*b1+(b1*c1-b2*c0)*x+(a2*c0-a1*c1)*y)
    else:
        out[:,0] = (a0+a1*x+a2*y) / (1+c0*x+c1*y)
        out[:,1] = (b0+b1*x+b2*y) / (1+c0*x+c1*y)
    return out

def make_polynomial(src, dst, n):
    '''
    Determine parameters of 2D polynomial transformation of order n,
    where the transformation is defined as:
        X = sum[j=0:n](sum[i=0:j](a_ji * x**(j-i)*y**i))
        Y = sum[j=0:n](sum[i=0:j](b_ji * x**(j-i)*y**i))
    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param src: :class:`numpy.array`
        Nx2 coordinate matrix of destination coordinate system

    :returns: params, None
    '''

    xs = src[:,0]
    ys = src[:,1]
    # number of unknown coefficients
    u = (n+1)*(n+2)
    rows = src.shape[0]
    A = np.zeros((rows*2, u))
    pidx = 0
    for j in xrange(n+1):
        for i in xrange(j+1):
            A[:rows,pidx] = xs**(j-i)*ys**i
            A[rows:,pidx+u/2] = xs**(j-i)*ys**i
            pidx += 1
    b = np.zeros((rows*2,))
    b[:rows] = dst[:,0]
    b[rows:] = dst[:,1]
    params = np.linalg.lstsq(A, b)[0]
    return params, None

def polynomial_transform(coords, params, inverse=False):
    '''
    Apply polynomial transformation.

    :param coords: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param params: :class:`numpy.array`
        parameters returned by `make_tform`
    :param inverse: bool
        apply inverse transformation, default is False

    :returns: :class:`numpy.array`
        transformed coordinates
    '''

    x = coords[:,0]
    y = coords[:,1]
    u = len(params)
    # number of coefficients ( -> u = (n+1)*(n+2) )
    n = int((-3+math.sqrt(9-4*(2-u))) / 2)
    out = np.zeros(coords.shape)
    if inverse:
        raise NotImplemented('There is no explicit way to do the inverse '
            'polynomial transformation as it is in general non-linear.'
            'Determine the inverse transformation parameters '
            'and use the fwd transformation instead.')
    else:
        pidx = 0
        for j in xrange(n+1):
            for i in xrange(j+1):
                out[:,0] += params[pidx]*x**(j-i)*y**i
                out[:,1] += params[pidx+u/2]*x**(j-i)*y**i
                pidx += 1
    return out

def make_affine(src, dst):
    '''
    Determine parameters of 2D or 3D affine transformation in the order:
        a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3
    where the transformation is defined as:
        X = a0 + a1*x + a2*y[ + a3*z]
        Y = b0 + b1*x + b2*y[ + b3*z]
        [Z = c0 + c1*x + c2*y + c3*z]
    You can determine the over-, well- and under-determined parameters
    with the least-squares method.
    Source and destination coordinates must be Nx2 or Nx3 matrices (x, y, z).

    Explicit parameters are in the order:
        a0, b0, c0, mx, my, mz, alpha [radians], beta [radians], gamma [radians]
    where the 3D transformation is defined as (excluding the :
        X = tx * R3(gamma)*R2(beta)*R1(alpha)*S*x
        with
            X = (X, Y, Z).T
            tx = (a0, b0, c0).T
            R1(alpha) = rotation_matrix(alpha, axis=1)
            R2(beta) = rotation_matrix(beta, axis=2)
            R3(gamma) = rotation_matrix(gamma, axis=3)
            S = diag(mx, my, mz)
            x = (x, y, z).T
    and the simplified 2D transformation as:
        X = a0 + mx*x*cos(alpha) - my*y*sin(alpha+beta)
        Y = b0 + mx*x*sin(alpha) + my*y*cos(alpha+beta)

    In case of 2D coordinates the following parameters are 0:
        a3, b3, c0, c1, c2, c3
    and the explicit parameters
        c0, mz, gamma

    :param src: :class:`numpy.array`
        Nx2 or Nx3 coordinate matrix of source coordinate system
    :param src: :class:`numpy.array`
        Nx2 or Nx3 coordinate matrix of destination coordinate system

    :returns: params, params_explicit
    '''

    xs = src[:,0]
    ys = src[:,1]
    rows = src.shape[0]
    A = np.zeros((rows*3, 12))
    A[:rows,0] = 1
    A[:rows,1] = xs
    A[:rows,2] = ys
    A[rows:rows*2,4] = 1
    A[rows:rows*2,5] = xs
    A[rows:rows*2,6] = ys
    A[rows*2:,8] = 1
    A[rows*2:,9] = xs
    A[rows*2:,10] = ys
    b = np.zeros((rows*3,))
    b[:rows] = dst[:,0]
    b[rows:rows*2] = dst[:,1]
    if src.shape[1] == 3:
        zs = src[:,2]
        A[:rows,3] = zs
        A[rows:rows*2,7] = zs
        A[rows*2:,11] = zs
        b[rows*2:] = dst[:,2]
    params = np.linalg.lstsq(A, b)[0]
    #: determine explicit parameters
    a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3 = params
    mx = math.sqrt(a1**2+b1**2+c1**2)
    my = math.sqrt(a2**2+b2**2+c2**2)
    mz = math.sqrt(a3**2+b3**2+c3**2)
    if src.shape[1] == 3 and 0 in (mx, my, mz):
        warnings.warn('One of your scale factors are 0, you should probably '
            'use a 2D instead of a 3D affine transformation.', RuntimeWarning)
    alpha = math.atan2(c2, c3)
    beta = math.atan2(-c1, math.sqrt(a1**2+b1**2))
    gamma = math.atan2(b1, a1)
    params_explicit = np.array([a0, b0, c0, mx, my, mz, alpha, beta, gamma])
    return params, params_explicit

def affine_transform(coords, params, inverse=False):
    '''
    Apply 2D or 3D affine transformation.

    :param coords: :class:`numpy.array`
        Nx2 coordinate matrix of source coordinate system
    :param params: :class:`numpy.array`
        parameters returned by `make_tform`
    :param inverse: bool
        apply inverse transformation, default is False

    :returns: :class:`numpy.array`
        transformed coordinates
    '''

    a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3 = params
    x = coords[:,0]
    y = coords[:,1]
    out = np.zeros(coords.shape)
    if coords.shape[1] == 2:
        z = 0
        if inverse:
            out[:,0] = (a2*(y-b0)-b2*(x-a0)) / (a2*b1-a1*b2)
            out[:,1] = (b1*(x-a0)-a1*(y-b0)) / (a2*b1-a1*b2)
        else:
            out[:,0] = a0+a1*x+a2*y
            out[:,1] = b0+b1*x+b2*y
    elif coords.shape[1] == 3:
        z = coords[:,2]
        if inverse:
            out[:,0] = (
                    (b2*c3-b3*c2)*x - (b2*c3-b3*c2)*a0
                    + ((z-c0)*b3-y*c3+b0*c3)*a2 - ((z-c0)*b2-y*c2+b0*c2)*a3
                ) / ((b2*c3-b3*c2)*a1 - (b1*c3-b3*c1)*a2 + (b1*c2-b2*c1)*a3)
            out[:,1] = -(
                    (b1*c3-b3*c1)*x - (b1*c3-b3*c1)*a0
                    + ((z-c0)*b3-y*c3+b0*c3)*a1 - ((z-c0)*b1-y*c1+b0*c1)*a3
                ) / ((b2*c3-b3*c2)*a1 - (b1*c3-b3*c1)*a2 + (b1*c2-b2*c1)*a3)
            out[:,2] = (
                    (b1*c2-b2*c1)*x - (b1*c2-b2*c1)*a0
                    + ((z-c0)*b2-y*c2+b0*c2)*a1 - ((z-c0)*b1-y*c1+b0*c1)*a2
                ) / ((b2*c3-b3*c2)*a1 - (b1*c3-b3*c1)*a2 + (b1*c2-b2*c1)*a3)
        else:
            out[:,0] = a0+a1*x+a2*y+a3*z
            out[:,1] = b0+b1*x+b2*y+b3*z
            out[:,2] = c0+c1*x+c2*y+c3*z
    return out


MFUNCS = {
    'similarity': make_similarity,
    'bilinear': make_bilinear,
    'projective': make_projective,
    'polynomial': make_polynomial,
    'affine': make_affine,
}
TFUNCS = {
    'similarity': similarity_transform,
    'bilinear': bilinear_transform,
    'projective': projective_transform,
    'polynomial': polynomial_transform,
    'affine': affine_transform,
}


class Transformation(object):

    def __init__(self, ttype, params, params_explicit=None):
        '''
        Create transformation which allows you to do forward and inverse
        transformation and view the transformation parameters.

        :param ttype: similarity, bilinear, projective, polynomial, affine
            transformation type
        :param params: :class:`numpy.array`
            transformation parameters
        :param params: :class:`numpy.array`
            explicit transformation parameters as
        '''

        self.ttype = ttype
        self.params = params
        self.params_explicit = params_explicit

    def fwd(self, coords):
        '''
        Apply forward transformation.

        :param coords: :class:`numpy.array`
            Nx2 or Nx3 coordinate matrix
        '''

        single = False
        if coords.ndim == 1:
            coords = np.array([coords])
            single = True
        result = TFUNCS[self.ttype](coords, self.params, inverse=False)
        if single:
            return result[0]
        return result

    def inv(self, coords):
        '''
        Apply inverse transformation.

        :param coords: :class:`numpy.array`
            Nx2 or Nx3 coordinate matrix
        '''

        single = False
        if coords.ndim == 1:
            coords = np.array([coords])
            single = True
        result = TFUNCS[self.ttype](coords, self.params, inverse=True)
        if single:
            return result[0]
        return result


def rotation_matrix(angle, dim=2, axis=None):
    '''
    Create a 2D or 3D rotation matrix.

    :param: int or float as radians
        angle of rotation
    :param dim: 2 or 3, optional
        dimension of rotation matrix, default is 2
    :param axis: 1, 2 or 3, optional
        rotation axis for 3D rotation, default is None

    :returns: 2x2 or 3x3 rotation matrix
    '''

    if dim == 2:
        R = [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    elif dim == 3:
        if axis == 1:
            R = [
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)],
            ]
        elif axis == 2:
            R = [
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ]
        elif axis == 3:
            R = [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
    return np.array(R)
