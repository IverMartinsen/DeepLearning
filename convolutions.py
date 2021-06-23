import numpy as np
from  numba import jit

def convolve2d(x, w, stride, mode = "valid"):
    '''
    Performs 2d convolution between x and w.
    w is rotated and output is similar to scipy.signal.convolve2d.

    Parameters
    ----------
    x : numpy.ndarray
        2-D array to be filtered.
    w : numpy.ndarray
        2-D filter.
    stride : int
        movement/stride.
    mode : str, optional
        Options are "full" and "valid".
        The default is "valid".

    Raises
    ------
    ValueError
        If unsupported mode.

    Returns
    -------
    y : numpy.ndarray
        2-D array of filtered values.

    '''
    filterheight, filterwidth = w.shape
    
    if mode == 'valid':
        x = x
        H, W = x.shape

    elif mode == 'full':
        p = filterheight - 1
        x = np.pad(x, p)

    else:
        raise ValueError
    
    H, W = x.shape
    
    H_out = (H - filterheight) // stride + 1
    W_out = (W - filterwidth) // stride + 1
    
    x_col = np.zeros([H_out * W_out, filterheight * filterwidth])

    w_col = np.flip(w.reshape(-1))

    for i in range(H_out):
       for j in range(W_out):
           patch = x[i*stride : i*stride + filterheight, j*stride : j*stride + filterwidth]
           x_col[i*W_out + j, :] = np.reshape(patch, -1)
    
    y_col = np.matmul(x_col, w_col)
    
    y = y_col.reshape(H_out, W_out)
    
    return y

def convolve3d(x, w, stride):
    '''
    Performs 3-D convolution between a tensor x (C x H x W)
    and a filter w (C x FH x FW).
    
    Parameters
    ----------
    x : numpy.ndarray
        3-D array to be filtered.
    w : numpy.ndarray
        3-D filter.
    stride : int
        stride.

    Returns
    -------
    y : numpy.ndarray
        2-D array of filtered values.

    '''
    C, H, W = x.shape
    C, filterheight, filterwidth = w.shape
    
    H_out = (H - filterheight) // stride + 1
    W_out = (W - filterwidth) // stride + 1
    
    x_col = np.zeros([H_out * W_out, C * filterheight * filterwidth])

    w_col = np.rot90(w, 2, axes = (1, 2)).reshape(-1)

    for i in range(H_out):
       for j in range(W_out):
           patch = x[..., i*stride : i*stride + filterheight, j*stride : j*stride + filterwidth]
           x_col[i*W_out + j, :] = np.reshape(patch, -1)
    
    y_col = np.matmul(x_col, w_col)
    
    y = y_col.reshape(H_out, W_out)
    
    return y

@jit(nopython = True)
def convolve4d(x, w, stride):
    '''
    Performs 3-D convolution between 3-D input x
    and 4-D filter w. Filters are not rotated.

    Parameters
    ----------
    x : numpy.ndarray
        3-D array to be filtered.
    w : numpy.ndarray
        4-D filter.
    stride : int
        stride.

    Returns
    -------
    y : numpy.ndarray
        3-D array of filtered values.

    '''
    C_in, H, W = x.shape
    C_out, C_in, filterheight, filterwidth = w.shape
    
    H_out = (H - filterheight) // stride + 1
    W_out = (W - filterwidth) // stride + 1
    
    x_col = np.zeros((H_out * W_out, C_in * filterheight * filterwidth))

    w_col = np.ascontiguousarray(w).reshape(C_out, -1)

    for i in range(H_out):
       for j in range(W_out):
           patch = x[..., i*stride : i*stride + filterheight, j*stride : j*stride + filterwidth]
           tmp = np.ascontiguousarray(patch)
           x_col[i*W_out + j, :] = tmp.reshape(-1)
        
    y_col = (x_col @ w_col.transpose()).transpose()
    
    y = y_col.reshape(C_out, H_out, W_out)
    
    return y

@jit(nopython = True)
def maxpool(x, filtersize, stride):
    '''
    Performs max pooling of a 2-D input.

    Parameters
    ----------
    x : numpy.ndarray
        2-D input.
    filtersize : tuple
        Size of pooling window.
    stride : int
        Pooling stride.

    Returns
    -------
    y : numpy.ndarray
        2-D array of pooled values.

    '''
    H, W = x.shape
    filterheight, filterwidth = filtersize
    
    H_out = (H - filterheight) // stride + 1
    W_out = (W - filterwidth) // stride + 1
    
    x_col = np.zeros((H_out * W_out, filterheight * filterwidth))

    for i in range(H_out):
       for j in range(W_out):
           patch = x[i*stride : i*stride + filterheight, j*stride : j*stride + filterwidth]
           tmp = np.ascontiguousarray(patch)
           x_col[i*W_out + j, :] = tmp.reshape(filterheight * filterwidth)
    
    y_col = np.zeros(H_out * W_out)
    
    for i in range(H_out * W_out):
        
        y_col[i] = np.max(x_col[i])
    
    y = y_col.reshape(H_out, W_out)
    
    return y