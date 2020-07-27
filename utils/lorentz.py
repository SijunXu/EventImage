import numpy as np
import uproot_methods
import awkward

def to_p4(x):
    """
    input: (pt, eta, phi, mass) in shape of (nb_evt, max_nb_particles, 4), zero padded
    output: TLorentzVectorArray in shape of (nb_evt, nb_particles,)
    """
    mask = x[:, :, 0] > 0
    n_particles = np.sum(mask, axis=1)
    pt = awkward.JaggedArray.fromcounts(n_particles, x[:, :, 0][mask])
    eta = awkward.JaggedArray.fromcounts(n_particles, x[:, :, 1][mask])
    phi = awkward.JaggedArray.fromcounts(n_particles, x[:, :, 2][mask])
    #mass = awkward.JaggedArray.fromcounts(n_particles, np.zeros(x[:, :, 2].shape)[mask])
    mass = awkward.JaggedArray.fromcounts(n_particles, x[:, :, 3][mask])
    p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, mass)
    return p4

