import numpy as np
from scipy.linalg import ordqz, svd, lu_factor, lu_solve

def qz_solve(Ai, Bi, Ci, Fi, Gi, N, neq, neps):
    # Constructing augmented matrices as per Fortran subroutine
    AA = np.zeros((2 * neq, 2 * neq), dtype=np.complex_)
    BB = np.zeros_like(AA)
    CC = np.zeros((2 * neq, neps), dtype=np.complex_)
 
    AA[:neq, neq:] = -Ai
    BB[:neq, :neq] = Ci
    BB[:neq, neq:] = Bi
    CC[:neq, :] = Gi + np.dot(Fi, N)
 
    for i in range(neq):
        AA[neq + i, i] = 1.0
        BB[neq + i, neq + i] = 1.0
 
    print(f'BB = {BB}',f'AA = {AA}')
    print(f'Ai = {Ai}',f'Bi = {Bi}',f'Ci = {Ci}',f'Fi = {Fi}',f'Gi = {Gi}',f'N = {N}')
    # QZ decomposition
    TTS, SSS, alpha, beta, QS, ZS = ordqz(BB, AA, output='complex', sort='iuc')
 
    # Additional computations based on Fortran code
    nunstab = np.sum(np.abs(beta/alpha) > 1)
 
    RC = 0
    if nunstab > neq:
        print('System is unstable')
        RC = 1
        return None, None, RC
    elif nunstab < neq:
        print('System is indeterminant')
        RC = 2
        return None, None, RC
 
    # Checking if the system's eigenvalues can be translated to initial conditions
    ZSupp = ZS[:neq, :neq]
    U, s, Vh = svd(ZSupp)
    rank = np.sum(s > 1e-13)
 
    if rank < neq:
        print('Untranslatable initial conditions')
        RC = 3
        return None, None, RC
 
    # Further calculations
    Gamma = np.zeros((neq, neps), dtype=np.complex_)
    QC = QS[neq:,:] @ CC
    for j in range(neq):
        r_tran = QC[neq-j-1,:]#np.dot(QS[neq + 1 - j:, :].T.conj(), CC)
        for i in range(neq + 2 - j, neq):
            r_tran += TTS[2 * neq - j - 1, i + neq-1] * Gamma[i, :]
            r_tran -= np.dot(SSS[2 * neq + 1 - j, i + neq].conj(), N) @ Gamma[i, :]
 
        Ni = SSS[2 * neq - j-1, 2 * neq - j-1] * N - TTS[2 * neq - j-1, 2 * neq  - j-1]
        lu, piv = lu_factor(Ni)
        Gamma[neq - j - 1, :] = lu_solve((lu, piv), r_tran)
 
    izs = np.linalg.inv(ZS[:neq, :neq])
    alpha_zs = np.dot(ZS[neq:, :neq], izs)
    beta_zs = np.dot(ZS[neq:, neq:] - alpha_zs @ ZS[:neq, neq:], Gamma)
 
    ralpha = np.real(alpha_zs)
    rbeta = np.real(beta_zs)

    if np.isnan(rbeta).any() or np.isnan(ralpha).any():
        print('NaN encountered in ralpha or rbeta')
        return None, None, RC
 
    return ralpha, rbeta, RC
