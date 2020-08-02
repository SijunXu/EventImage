import multiprocessing
from functools import partial
from contextlib import contextmanager
import numpy as np
from pyjet import cluster, DTYPE_PTEPM

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def cluster_jet_single(data, nb_jets):
    """
    cluster jets with ee-kt algorithm
    """
    data = data.astype(np.float64)
    idx = []
    for j in range(len(data)):
        if data[j, 0]>0:
            idx.append(j)
    pseudojets_input = np.zeros(len(idx), dtype=DTYPE_PTEPM)
    for j, p_idx in enumerate(idx):
        if (data[p_idx, 0]>0):
            pseudojets_input[j]['pT'] = data[p_idx, 0]
            pseudojets_input[j]['eta'] = data[p_idx, 1]
            pseudojets_input[j]['phi'] = data[p_idx, 2]
            pseudojets_input[j]['mass'] = data[p_idx, 3]            
    sequence = cluster(pseudojets_input, algo='ee_kt')
    jets = sequence.exclusive_jets(nb_jets)
    event = np.zeros(nb_jets*4)
    if nb_jets == len(jets):
        for n_j in range(nb_jets):
            event[n_j*4:(n_j+1)*4] = np.array([jets[n_j].pt, jets[n_j].eta, jets[n_j].phi, jets[n_j].mass])
    return event

def cluster_data(X, nb_jets=2, n_jobs=-1):
    if n_jobs==-1:
        n_jobs = multiprocessing.cpu_count()
    with poolcontext(processes=n_jobs) as pool:
        jet = pool.map(partial(cluster_jet_single, nb_jets=nb_jets), X)
    jet = np.asarray(jet)
    return jet


