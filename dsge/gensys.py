# pure python implementation of GENSYS by Chris Sims
import warnings
import numpy as np
from scipy.linalg import ordqz, svd


def gensys(
    G0,
    G1,
    PSI,
    PI,
    C0=None,
    DIV=1 + 1e-6,
    REALSMALL=1e-6,
    return_everything=False,
    return_diagnostics: bool = False,
):
    """
    Solves a Linear Rational Expectations model via GENSYS.

    Γ₀xₜ = Γ₁xₜ₋₁ + Ψεₜ + Πηₜ

    Returns
    -------

    RC : 2d int array
         [ 1,  1] = existence and uniqueness
         [ 1,  0] = existence, not uniqueness
         [-2, -2] = coincicdent zeros

    Notes
    -----
    The solution method is detailed in ...

    """
    n = G0.shape[0]

    # Backward compat: some call sites pass a scalar DIV as the 5th arg.
    if C0 is not None:
        c0 = np.asarray(C0)
        if c0.ndim == 0:
            DIV = float(c0)
            C0 = None

    def _eig_select(alpha, beta):
        with np.errstate(divide="ignore", invalid="ignore"):
            x = beta / alpha
        return np.abs(x) < DIV

    with np.errstate(invalid='ignore', divide='ignore'):
        AA, BB, alpha, beta, Q, Z = ordqz(G0, G1, sort=_eig_select, output='complex')
        # "Coincident zeros" (0/0 generalized eigenvalues) can be falsely detected if we use an
        # absolute threshold on large, poorly-scaled systems. Use a relative scale based on the
        # QZ output magnitudes instead.
        alpha_scale = float(max(1.0, np.max(np.abs(alpha)))) if np.size(alpha) else 1.0
        beta_scale = float(max(1.0, np.max(np.abs(beta)))) if np.size(beta) else 1.0
        zxz = ((np.abs(beta) < REALSMALL * beta_scale) & (np.abs(alpha) < REALSMALL * alpha_scale)).any()

        x = beta / alpha
        nstable = int(np.sum(np.abs(x) < DIV))
        nunstab = n - nstable

    if zxz:
        RC = np.array([-2, -2], dtype=int)
        diag = {
            "eig": x,
            "nstable": nstable,
            "nunstable": nunstab,
            "coincident_zeros": True,
            "alpha_scale": alpha_scale,
            "beta_scale": beta_scale,
        }
        if return_diagnostics:
            return None, None, RC, diag
        return None, None, RC
        
    Q = Q.T.conjugate()
    Qstab, Qunstab = Q[:nstable, :], Q[nstable:, :]

    RC = np.array([0, 0])
    tol = float(REALSMALL)

    if nunstab == 0:
        ueta = np.zeros((0, 0))
        sv_unstable = np.array([], dtype=float)
        veta = np.zeros((PI.shape[1], 0))
        RC[0] = 1
    else:
        etawt = Qunstab.dot(PI)
        ueta, sv_unstable_all, veta = svd(etawt, full_matrices=False)

        tol = float(max(REALSMALL, REALSMALL * float(np.max(sv_unstable_all)) if np.size(sv_unstable_all) else REALSMALL))
        bigev = sv_unstable_all > tol
        sv_unstable = sv_unstable_all[bigev]
        ueta = ueta[:, bigev]
        veta = veta[bigev, :].conjugate().T

        RC[0] = int(sv_unstable.size >= nunstab)

    if RC[0] == 0:
        warnings.warn(
            f"{nunstab} unstable roots, but only {sv_unstable.size} "
            " RE errors! No solution.")

    if nunstab == n:
        raise NotImplementedError("case nunstab == n, not implemented")
    else:
        etawt1 = Qstab.dot(PI)
        ueta1, deta1, veta1 = svd(etawt1, full_matrices=False)
        tol1 = float(max(REALSMALL, REALSMALL * float(np.max(deta1)) if np.size(deta1) else REALSMALL))
        bigev = deta1 > tol1
        sv_stable = deta1[bigev]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[bigev, :].conjugate().T

    dl_loose = np.array([], dtype=float)
    if veta1.size == 0:
        unique = 1
        sv_stable = np.array([], dtype=float)
    else:
        loose = veta1 - veta.dot(veta.conjugate().T).dot(veta1)
        dl = svd(loose, compute_uv=False)
        dl_loose = np.asarray(dl, dtype=float)
        tol_unique = REALSMALL
        unique = bool((dl < tol_unique).all())

    if nunstab > 0:
        AA22 = AA[-nunstab:, :][:, -nunstab:]
        BB22 = BB[-nunstab:, :][:, -nunstab:]
        M = np.linalg.inv(BB22).dot(AA22)
    else:
        M = np.zeros((0, 0))

    if unique:
        RC[1] = 1
    else:
        warnings.warn("Indeterminacy: multiple stable solutions satisfy the model.")

    inv_sv_unstable = np.diag(1.0 / sv_unstable) if sv_unstable.size else np.zeros((0, 0))
    sv_stable_diag = np.diag(sv_stable) if sv_stable.size else np.zeros((0, 0))

    etawt_inverseT = ueta.dot((veta.dot(inv_sv_unstable)).conjugate().T)
    etatw1_T = veta1.dot(sv_stable_diag).dot(ueta1.conjugate().T)
    tmat = np.c_[np.eye(nstable), -(etawt_inverseT.dot(etatw1_T)).conjugate().T]

    G0 = np.r_[tmat.dot(AA), np.c_[np.zeros(
        (nunstab, nstable)), np.eye(nunstab)]]
    G1 = np.r_[tmat.dot(BB), np.zeros((nunstab, n))]

    G0i = np.linalg.inv(G0)
    G1 = G0i.dot(G1)

    impact = G0i.dot(np.r_[tmat.dot(Q).dot(
        PSI), np.zeros((nunstab, PSI.shape[1]))])

    G1 = np.real(Z.dot(G1).dot(Z.conjugate().T))
    impact = np.real(Z.dot(impact))



    diag = {
        "eig": x,
        "nstable": nstable,
        "nunstable": nunstab,
        "coincident_zeros": False,
        "existence": bool(RC[0]),
        "unique": bool(RC[1]),
        "sv_unstable": sv_unstable,
        "sv_stable": sv_stable,
        "sv_unstable_tol": tol if nunstab else REALSMALL,
        "sv_stable_tol": tol1 if veta1.size else REALSMALL,
        "unique_tol": REALSMALL,
        "sv_loose": dl_loose,
        "alpha_scale": alpha_scale,
        "beta_scale": beta_scale,
    }

    if return_everything:
        if nunstab > 0:
            GZ = -np.linalg.inv(BB22).dot(Qunstab).dot(PSI)
            GY = Z.dot(G0i[:, -nunstab:])
        else:
            GZ = np.zeros((0, PSI.shape[1]))
            GY = np.zeros((n, 0))

        if return_diagnostics:
            return G1, impact, M, GZ, GY, RC, diag
        return G1, impact, M, GZ, GY, RC

    if return_diagnostics:
        return G1, impact, RC, diag
    return G1, impact, RC
    # GZ is
