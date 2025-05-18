# cython: language_level=3

cimport cython

import numpy as np
cimport numpy as cnp

from libc.math cimport log2, exp


EPS = np.finfo(np.float32).eps


cdef class _Node:
    cdef public bint is_leaf
    cdef public object indices
    cdef public object prediction
    cdef public int feature
    cdef public double value
    cdef public _Node left
    cdef public _Node right

    def __init__(self, cnp.ndarray indices):
        self.is_leaf = True
        self.indices = indices
        self.prediction = None

    cpdef void split(self, int feature, double value, _Node left, _Node right):
        self.is_leaf = False
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double entropy(
    const cnp.int32_t[:] C,
    const cnp.int32_t[:] indices
):
    cdef:
        Py_ssize_t i, idx
        int label, total = indices.shape[0]
        int[2] counts = [0, 0]
        double entropy = 0., p

    if total == 0:
        return 0.

    for i in range(total):
        idx = indices[i]
        label = C[idx]
        counts[label] += 1

    for i in range(2):
        if counts[i] > 0:
            p = <double>(counts[i]) / total
            entropy -= p * log2(p)

    return entropy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple best_split(
    cnp.ndarray[double, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] C,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    double alpha,
    double beta
):
    cdef int n_samples = indices.shape[0]
    cdef int n_features = X.shape[1]
    cdef int i, s, feature
    cdef double value, criterion, best_criterion = 0.
    cdef cnp.int32_t[:] left, right
    cdef cnp.int32_t[:] sorted_indices = np.empty(n_samples, dtype=np.int32)
    cdef double left_entropy, right_entropy, total_entropy
    cdef cnp.int32_t[:] best_left = None
    cdef cnp.int32_t[:] best_right = None
    cdef int best_feature
    cdef double best_value

    total_entropy = entropy(C, indices)
    for feature in range(n_features):
        sorted_indices = indices[np.argsort(X[indices, feature])]
        s = 1
        for i in range(n_samples - 1):
            if X[sorted_indices[i], feature] == X[sorted_indices[i + 1], feature]:
                continue
            if s > 1:
                s -= 1
                continue
            value = (X[sorted_indices[i], feature] + X[sorted_indices[i + 1], feature]) / 2
            left, right = sorted_indices[:i + 1], sorted_indices[i + 1:]
            left_entropy, right_entropy = entropy(C, left), entropy(C, right)
            criterion = total_entropy - (left_entropy * len(left) + right_entropy * len(right)) / n_samples
            if criterion > best_criterion:
                best_criterion, best_feature, best_value, best_left, best_right = \
                    criterion, feature, value, left, right
            s = int((alpha * n_samples) / (1. + exp(beta * (criterion / (best_criterion + EPS) - .5))))

    return best_feature, best_value, np.array(best_left, dtype=np.int32), np.array(best_right, dtype=np.int32)
