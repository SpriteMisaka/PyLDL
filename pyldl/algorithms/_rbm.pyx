# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as cnp

from libc.math cimport exp


cdef sigmoid(cnp.ndarray[double, ndim=2] Z):
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[double, ndim=2] out = np.empty_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            out[i, j] = 1.0 / (1.0 + exp(-Z[i, j]))
    return out


cdef average_diff(
    cnp.ndarray[double, ndim=2] Z0,
    cnp.ndarray[double, ndim=2] Z1
):
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[double, ndim=1] dZ = np.zeros(Z0.shape[1], dtype=np.float64)
    for i in range(Z0.shape[0]):
        for j in range(Z0.shape[1]):
            dZ[j] += (Z0[i, j] - Z1[i, j])
    dZ /= Z0.shape[0]
    return dZ


cdef energy(
    cnp.ndarray[double, ndim=2] X,
    cnp.ndarray[double, ndim=2] H,
    cnp.ndarray[double, ndim=2] W,
    cnp.ndarray[double, ndim=1] b,
    cnp.ndarray[double, ndim=1] c
):
    cdef double e = 0.0
    e -= np.sum(X @ b.reshape(-1, 1))
    e -= np.sum(H @ c.reshape(-1, 1))
    e -= np.sum((X @ W) * H)
    return e


cpdef train_rbm(
    cnp.ndarray[double, ndim=2] X,
    cnp.ndarray[double, ndim=2] W,
    cnp.ndarray[double, ndim=1] b,
    cnp.ndarray[double, ndim=1] c,
    int iterations, int batch_size,
    double lr, double init_t, double final_t
):
    cdef int iter, start, end
    cdef double dE, t

    cdef cnp.ndarray[double, ndim=2] X0, X1, H0, H1, H0s, H1s
    cdef cnp.ndarray[double, ndim=2] randH, randH1

    for iter in range(iterations):
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            X0 = X[start:end]
            H0 = sigmoid(X0 @ W + c)
            randH = np.random.rand(H0.shape[0], H0.shape[1])
            H0s = (H0 > randH).astype(np.float64)

            X1 = sigmoid(H0s @ W.T + b)
            H1 = sigmoid(X1 @ W + c)
            randH1 = np.random.rand(H1.shape[0], H1.shape[1])
            H1s = (H1 > randH1).astype(np.float64)

            dE = energy(X0, H0s, W, b, c) - energy(X1, H1s, W, b, c)
            t = max(final_t, init_t * pow(0.7, iter))
            if dE < 0 or np.random.rand() < np.exp(-dE / t):
                X0 = X1

            W += lr * ((X0.T @ H0s - X1.T @ H1s) / X0.shape[0])
            b += lr * average_diff(X0, X1)
            c += lr * average_diff(H0s, H1s)

    H = sigmoid(X @ W + c)
    return H, W, b, c
