# pure python implementation of GENSYS by Chris Sims
import warnings
import numpy as np
from scipy.linalg import ordqz, svd

def gensys(G0, G1, PSI, PI, DIV=1 + 1e-8,
           REALSMALL=1e-6,
           return_everything=False):
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
    n, pin = G0.shape[0], PI.shape[1]

    with np.errstate(invalid='ignore', divide='ignore'):
        AA, BB, alpha, beta, Q, Z = ordqz(G0, G1, sort='ouc', output='complex')
        zxz = ((np.abs(beta) < REALSMALL) * (np.abs(alpha) < REALSMALL)).any()

        x = alpha / beta
        nunstab = (x * x.conjugate() < 1.0).sum()

    if zxz:
        RC = [-2, -2]
        print("Coincident zeros")
        return

    nstab = n - nunstab

    Q = Q.T.conjugate()
    Qstab, Qunstab = Q[:nstab, :], Q[nstab:, :]

    etawt = Qunstab.dot(PI)
    ueta, deta, veta = svd(etawt, full_matrices=False)

    bigev = deta > REALSMALL
    deta = deta[bigev]
    ueta = ueta[:, bigev]
    veta = veta[bigev, :].conjugate().T

    RC = np.array([0, 0])
    RC[0] = len(bigev) >= nunstab

    if RC[0] == 0:
        warnings.warn(
            f"{nunstab} unstable roots, but only {len(bigev)} "
            " RE errors! No solution.")

    if nunstab == n:
        raise NotImplementedError("case nunstab == n, not implemented")
    else:
        etawt1 = Qstab.dot(PI)
        ueta1, deta1, veta1 = svd(etawt1, full_matrices=False)
        bigev = deta1 > REALSMALL
        deta1 = deta1[bigev]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[bigev, :].conjugate().T

    if veta1.size == 0:
        unique = 1
    else:
        loose = veta1 - veta.dot(veta.conjugate().T).dot(veta1)
        ul, dl, vl = svd(loose)
        unique = (dl < REALSMALL).all()

        # existence for general epsilon[t]
        AA22 = AA[-nunstab:, :][:, -nunstab:]
        BB22 = BB[-nunstab:, :][:, -nunstab:]
        M = np.linalg.inv(BB22).dot(AA22)

    if unique:
        RC[1] = 1
    else:
        pass
        # print("Indeterminancy")

    deta = np.diag(1.0 / deta)
    deta1 = np.diag(deta1)

    etawt_inverseT = ueta.dot((veta.dot(deta)).conjugate().T)
    etatw1_T = veta1.dot(deta1).dot(ueta1.conjugate().T)
    tmat = np.c_[np.eye(nstab), -(etawt_inverseT.dot(etatw1_T)).conjugate().T]

    G0 = np.r_[tmat.dot(AA), np.c_[np.zeros(
        (nunstab, nstab)), np.eye(nunstab)]]
    G1 = np.r_[tmat.dot(BB), np.zeros((nunstab, n))]

    G0i = np.linalg.inv(G0)
    G1 = G0i.dot(G1)

    impact = G0i.dot(np.r_[tmat.dot(Q).dot(
        PSI), np.zeros((nunstab, PSI.shape[1]))])

    G1 = np.real(Z.dot(G1).dot(Z.conjugate().T))
    impact = np.real(Z.dot(impact))

    if return_everything:
        GZ = -np.linalg.inv(BB22).dot(Qunstab).dot(PSI)
        GY = Z.dot(G0i[:, -nunstab:])

        return G1, impact, M, GZ, GY, RC

    else:
        return G1, impact, RC
