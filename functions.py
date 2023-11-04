import numpy as np
import pandas as pd
import tensorly.backend as T
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.base import unfold, fold
import tensorly.tenalg
import tensorly.random
from scipy import linalg
from .svd import svd_fun


def svd_init(tensor, modes, ranks):
    factors = []
    for index, mode in enumerate(modes):
        eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=ranks[index])
        factors.append(eigenvecs)
    return factors


def init(dims, ranks):
    factors = []
    for index, rank in enumerate(ranks):
        U_i = np.zeros((rank, dims[index]))
        mindim = min(dims[index], rank)
        for i in range(mindim):
            U_i[i][i] = 1
        factors.append(U_i)
    return factors


def autocorr(Y, lag=10):
    T = len(Y)
    r = []
    for l in range(lag+1):
        product = 0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(Y[t] * Y[tl])
        r.append(product)
    return r


def fit_ar(Y, p=10):
    r = autocorr(Y, p)
    R = linalg.toeplitz(r[:p])
    r = r[1:]
    A = linalg.pinv(R).dot(r)
    return A



