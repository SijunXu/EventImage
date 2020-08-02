import multiprocessing
from functools import partial
from contextlib import contextmanager
import numpy as np
from scipy.special import legendre

from .lorentz import to_p4

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def FW_single(X_4p, lmax, center_of_mass=240.):
    """
    calculate the first lmax FW-moments (l<=lmax-1)
    input: X_4p -> lorentz vector of an event
    """
    FW_moments = np.zeros(lmax)
    weight = np.zeros((len(X_4p), len(X_4p)))
    omega = np.zeros(weight.shape)
    for i, p in enumerate(X_4p):
        for j, q in enumerate(X_4p):
            weight[i, j] = p.E * q.E/np.square(center_of_mass) #weighted with center of energy
            omega[i, j] = np.cos(p.theta) * np.cos(q.theta) + np.sin(p.theta) * np.sin(q.theta) * np.cos(p.phi-q.phi)
            if omega[i, j] > 1.:
                omega[i, j] = 1.
            elif omega[i, j] < -1.:
                omega[i, j] = -1.
    for l in range(lmax):
        FW_moments[l] = np.sum(np.multiply(weight, legendre(l)(omega)))
    return FW_moments

def FW_parallel(X, n_jobs=-1, **kwargs):
    """
    computs FW-moments at particle-level.
    input: X -> the zero padded of 4-momenta of particles, in shape of (n_evt, n_max_par, (pt, eta, phi, mass))
           n_jobs: number of cores to run; using all cores if n_jobs=-1
           kwargs: (lmax->int, center_of_mass->float)
    returns: fw in shape of (n_evt, lmax)
    """
    X_4p = to_p4(X)
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    with poolcontext(processes=n_jobs) as pool:
        fw = pool.map(partial(FW_single, **kwargs), X_4p)
    return np.asarray(fw, dtype=np.float64)
