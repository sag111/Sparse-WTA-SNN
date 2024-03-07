import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class Pooling(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 input_shape,
                 kernel_size: tuple = (2,2), 
                 stride: tuple = (2,2), 
                 method: str = 'max', 
                 pad: bool = False):
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.method = method
        self.pad = pad
        self.input_shape = input_shape
        assert len(self.input_shape) > 1, "Input vectors must be at least 2-dimensional."

    def _as_stride(self, arr):
        """Get a strided sub-matrices view of an ndarray.
        See also skimage.util.shape.view_as_windows()
        """
        s0, s1 = arr.strides[-2:]
        m1, n1 = arr.shape[-2:]
        m2, n2 = self.kernel_size
        view_shape = arr.shape[:-2] + (
            1 + (m1 - m2) // self.stride[0],
            1 + (n1 - n2) // self.stride[1],
            m2,
            n2,
        )
        strides = arr.strides[:-2] + (self.stride[0] * s0, self.stride[1] * s1, s0, s1)
        subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
        return subs


    def pooling(self, mat):

        m, n = mat.shape[-2:]
        ky, kx = self.kernel_size
        if self.stride is None:
            self.stride = (ky, kx)
        sy, sx = self.stride

        _ceil = lambda x, y: int(np.ceil(x / float(y)))

        if self.pad:
            ny = _ceil(m, sy)
            nx = _ceil(n, sx)
            size = mat.shape[:-2] + ((ny - 1) * sy + ky, (nx - 1) * sx + kx)
            mat_p = np.full(size, np.nan)
            mat_p[..., :m, :n] = mat
        else:
            mat_p = mat[..., : (m - ky) // sy * sy + ky, : (n - kx) // sx * sx + kx]

        view = self._as_stride(mat_p)

        if self.method == "max":
            result = np.nanmax(view, axis=(-2, -1))
        else:
            result = np.nanmean(view, axis=(-2, -1))

        return result

    def fit(self, X, y=None):
        X = check_array(X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X):

        X = X.reshape((-1, *self.input_shape)) # convert to 2d
        batch_size = X.shape[0]

        X = self.pooling(X)

        return X.reshape((batch_size, -1)) # flatten back
