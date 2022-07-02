import numpy as np
import xarray as xr


def disp_from_mt(
        mt_symmat, cosine_vecs, far=True, intermed=True, near=True):
    """
    Radiation patterns for displacement field due to an arbitrary
    moment-tensor point source.

    Parameters
    ----------
    mt_symmat : ndarray of shape (3, 3)
        Seismic moment tensor as a plain *symmetric* matrix.
    cosine_vecs : ndarray of shape (n_receivers, 3)
        Unit vectors of direction cosines.

    Returns
    -------
    ds : :py:class:`xarray.Dataset` object
        Radiation patterns for displacement due to a moment-tensor point
        source. The Dataset keys are `{'FP', 'FS', 'IP', 'IS', 'N'}`.
        Each key is mapped to a 2-D array, whose shape is
        `(n_receivers, 3)` and dimesion names are `'i_receiver'` and
        `'axis'`, with sizes equal to `n_receivers` and 3, respectively.
        The coordinate names of the dimension `axis` are `{'x', 'y', 'z'}`
        corresponding to indices 0, 1 and 2, respectively.
    """
    # M.shape=(3, 3), Γ.shape=(n_rec, 3, 1), ΓT.shape=(n_rec, 1, 3)
    M = np.asarray(mt_symmat)
    Γ = cosine_vecs[..., np.newaxis]
    ΓT = Γ.transpose((0, 2, 1))

    # Cache common terms to save computation time
    q1 = M @ Γ
    k1 = ΓT @ q1   # shape of (n_rec, 1, 1)
    k2 = M.trace()
    q2 = k1 * Γ
    q3 = k2 * Γ

    # Far-field displacements (FP and FS ∝ 1/r)
    if far is True:
        A_fp = q2
        A_fs = q2 - q1
    else:
        A_fp = A_fs = np.zeros_like(Γ)

    # Intermediate-field displacement (IP and IS ∝ 1/r²)
    if intermed is True:
        A_ip = (6.0 * q2) - q3 - (2.0 * q1)
        A_is = (6.0 * q2) - q3 - (3.0 * q1)
    else:
        A_ip = A_is = np.zeros_like(Γ)

    # Near-field displacement (N ∝ 1/r⁴)
    if near is True:
        A_n = 3.0 * ((5.0 * q2) - q3 - (2.0 * q1))
    else:
        A_n = np.zeros_like(Γ)

    # Dimension and coordinate names when saving RPs into `xr.DataSet`
    dims = ['i_receiver', 'axis']
    coords = {'axis': ['x', 'y', 'z']}

    # Remove last axis, (n_rec, 3, 1) -> (n_rec, 3), then save
    data_vars = {
        k: (dims, v.squeeze(axis=2))
        for k, v in zip(
            ['FP', 'FS', 'IP', 'IS', 'N'],
            [A_fp, A_fs, A_ip, A_is, A_n])}

    return xr.Dataset(data_vars=data_vars, coords=coords)


def disp_from_sf(force_vec, cosine_vecs, far=True, near=True):
    """
    Parameters
    ----------
    force_vec : ndarray of shape (3, 1)
        Vector of the single-force components ordered like (F1, F2, F3).
    cosine_vecs : ndarray of shape (n_receivers, 3)
        Unit vectors of direction cosines.

    Returns
    -------
    ds : :py:class:`xarray.Dataset` object
        Radiation patterns for displacement due to a single-force point
        source. The Dataset keys are `{'FP', 'FS', 'N'}`. Each key is
        mapped to a 2-D array, whose shape is `(n_receivers, 3)` and
        dimesion names are `'i_receiver'` and `'axis'`, with sizes equal
        to `n_receivers` and 3, respectively. The coordinate names of
        the dimension `axis` are `{'x', 'y', 'z'}` corresponding to
        indices 0, 1 and 2, respectively.
    """
    # Array shapes are F::(3, 1), Γ::(n_rec, 3, 1)
    F = np.asarray(force_vec)
    Γ = cosine_vecs[..., np.newaxis]

    # Cache common terms to save computation time
    q = (F.T @ Γ) * Γ

    # Far-field displacement (FP and FS ∝ 1/r)
    if far is True:
        A_fp = q
        A_fs = -q + F
    else:
        A_fp = A_fs = np.zeros_like(Γ)

    # Near-field displacement (N ∝ 1/r³)
    if near is True:
        A_n = 3.0 * q - F
    else:
        A_n = np.zeros_like(Γ)

    # Dimension and coordinate names when saving RPs into `xr.DataSet`
    dims = ['i_receiver', 'axis']
    coords = {'axis': ['x', 'y', 'z']}

    # Remove last axis, (n_rec, 3, 1) -> (n_rec, 3), then save
    data_vars = {
        k: (dims, v.squeeze(axis=2))
        for k, v in zip(['FP', 'FS', 'N'], [A_fp, A_fs, A_n])}

    return xr.Dataset(data_vars=data_vars, coords=coords)


def normal_strain_from_mt(
        mt_symmat, cosine_vecs, far=True, intermed_far=True,
        intermed_near=True, near=True):
    """
    Radiation patterns for *normal* strain field (εᵢᵢ; i∈{x, y, z}) due
    to an arbitrary moment-tensor point source.

    Parameters
    ----------
    mt_symmat: ndarray of shape (3, 3)
        Seismic moment tensor as a plain *symmetric* matrix.
    cosine_vecs : ndarray of shape (n_receivers, 3)
        Unit vectors of direction cosines.

    Returns
    -------
    ds : :py:class:`xarray.Dataset` object
        Radiation patterns for strain due to a moment-tensor
        point source. The Dataset keys are
        `{'FP', 'FS', 'IFP', 'IFS', 'INP', 'INS', 'N'}`. Each key is
        mapped to a 2-D array, whose shape is `(n_receivers, 3)` and
        dimesion names are `'i_receiver'`, and `'axis'`, with sizes
        equal to `n_receivers` and 3, respectively. The coordinate names
        of the dimension `axis` are `{'x', 'y' and 'z'}`, corresponding
        to indices 0, 1 and 2, respectively.
    """
    # Array shapes are M::(3, 3), Γ::(n_rec, 3, 1), ΓT::(n_rec, 1, 3)
    M = np.asarray(mt_symmat)
    Γ = cosine_vecs[..., np.newaxis]
    ΓT = Γ.transpose((0, 2, 1))

    # Cache common terms to save computation time
    q1 = M @ Γ
    k1 = ΓT @ q1   # shape of (n_rec, 1, 1)
    k2 = M.trace()
    q2 = k1 * Γ
    q3 = k2 * Γ
    J = np.ones((3, 1), dtype=np.float64)
    q4 = k1 * J
    q5 = k2 * J
    q6 = M.diagonal()[:, np.newaxis]

    # Far-field strain (FP, FS ∝ 1/r)
    if far is True:
        A_fp = q2
        B_fp = A_fp * Γ

        A_fs = q2 - q1
        B_fs = A_fs * Γ
    else:
        B_fp = B_fs = np.zeros_like(Γ)

    # Intermediate-far field strain (IFP and IFS ∝ 1/r²)
    if intermed_far is True:
        A_ip = (6.0 * q2) - q3 - (2.0 * q1)
        B_ifp = q4 + ((-4.0 * q2) + (2.0 * q1) - A_ip) * Γ

        A_is = (6.0 * q2) - q3 - (3.0 * q1)
        B_ifs = q4 + ((-4.0 * q2) + (4.0 * q1) - A_is) * Γ - q6
    else:
        B_ifp = B_ifs = np.zeros_like(Γ)

    # Intermediate-near field strain (INP and INS ∝ 1/r³)
    if intermed_near is True:
        A_n = 3.0 * ((5.0 * q2) - q3 - (2.0 * q1))
        B_inp = (
            (6.0 * q4)
            + ((-30.0 * q2) + (18.0 * q1) + (3.0 * q3) - A_n) * Γ
            - q5 - (2.0 * q6))

        B_ins = (
            (6.0 * q4)
            + ((-30.0 * q2) + (21.0 * q1) + (3.0 * q3) - A_n) * Γ
            - q5 - (3.0 * q6))
    else:
        B_inp = B_ins = np.zeros_like(Γ)

    # Near-field strain (N ∝ 1/r⁵)
    if near is True:
        B_n = (
            (15.0 * q4) + 15.0 * ((-7.0 * q2) + (4.0 * q1) + q3) * Γ
            - (3.0 * q5) - (6.0 * q6))
    else:
        B_n = np.zeros_like(Γ)

    # Dimension and coordinate names when saving RPs into `xr.DataSet`
    dims = ['i_receiver', 'axis']
    coords = {'axis': ['x', 'y', 'z']}

    # Remove last axis, (n_rec, 3, 1) -> (n_rec, 3), then save
    data_vars = {
        k: (dims, v.squeeze(axis=2))
        for k, v in zip(
            ['FP', 'FS', 'IFP', 'IFS', 'INP', 'INS', 'N'],
            [B_fp, B_fs, B_ifp, B_ifs, B_inp, B_ins, B_n])}

    return xr.Dataset(data_vars=data_vars, coords=coords)


__all__ = ['disp_from_mt', 'disp_from_sf', 'normal_strain_from_mt']
