from __future__ import division

import numpy as np
import scipy as sp
import pandas as p
from fortran import dlyap, kalman, gensys



class StateSpaceModel(object):

    def __init__(self, yy, TT, RR, QQ, DD, ZZ, HH, t0=0,
                 shock_names=None, state_names=None, obs_names=None):

        if len(yy.shape) < 2:
            yy = np.swapaxes(np.atleast_2d(yy), 0, 1)

        self.yy = yy


        self.TT = TT
        self.RR = RR
        self.QQ = QQ
        self.DD = DD
        self.ZZ = ZZ
        self.HH = HH


        self.t0 = t0

        self.shock_names = None

    def log_lik(self, para, *args, **kwargs):

        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0, info = dlyap.dlyap(TT, RR.dot(QQ).dot(RR.T))

        lik = kalman.kalman_filter(yy.T, TT, RR, QQ, DD, ZZ, HH, P0)

        return lik

    def log_lik_pf(self, para, *args, **kwargs):

        t0 = kwargs.pop('t0', self.t0)
        npart = kwargs.pop('npart', 1000)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0, info = dlyap.dlyap(TT, RR.dot(QQ).dot(RR.T))

        nobs = yy.shape[0]
        ny = yy.shape[1]
        neps = QQ.shape[0]
        ns = TT.shape[0]

        RRcQQ = RR.dot(np.linalg.cholesky(QQ))

        St = np.zeros((nobs+1, ns, npart))
        St[0, :, :] = np.linalg.cholesky(P0).dot(np.random.normal(size=(neps, npart)))

        resamp = np.zeros((nobs))
        for t in range(nobs):
            eps = RRcQQ.dot(np.random.normal(size=(neps, npart)))
            St[t+1, :, :] = TT.dot(St[t, :, :]) + eps

            nut = np.tile(yy[t, :] - DD, (1, npart)) - ZZ.dot(St[t+1, :, :])




    def system_matrices(self, para, *args, **kwargs):
        TT = np.atleast_2d(self.TT(para, *args, **kwargs))
        RR = np.atleast_2d(self.RR(para, *args, **kwargs))
        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))

        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        return TT, RR, QQ, DD, ZZ, HH


    def impulse_response(self, para, h=20, *args, **kwargs):

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if self.shock_names==None:
            self.shock_names = ['shock_' + str(i) for i in range(QQ.shape[0])]
            self.state_names = ['state_' + str(i) for i in range(TT.shape[0])]
            self.obs_names = ['obs_' + str(i) for i in range(ZZ.shape[0])]


        neps = QQ.shape[0]

        irfs = {}
        for i in range(neps):

            At = np.zeros((TT.shape[0], h+1))
            QQz = np.zeros_like(QQ)
            QQz[i, i] = QQ[i, i]
            cQQz = np.sqrt(QQz)

            #cQQz = np.linalg.cholesky(QQz)

            At[:, 0] = (RR.dot(cQQz)[:, i]).squeeze()

            for j in range(h):
                At[:, j+1] = TT.dot(At[:, j])

            irfs[self.shock_names[i]] = p.DataFrame(At.T, columns=self.state_names)

        return p.Panel(irfs)

    def simulate(self, para, nsim, *args, **kwargs):

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        ysim  = np.zeros((nsim*2, DD.size))

        At = np.zeros((TT.shape[0],))

        for i in range(nsim*2):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = TT.dot(At) + RR.dot(e)

            ysim[i, :] = DD.T + ZZ.dot(At)

        return ysim[nsim:, :]


    def forecast(self, para, h=20, *args, **kwargs):
        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0, info = dlyap.dlyap(TT, RR.dot(QQ).dot(RR.T))

        data = np.asarray(yy)
        nobs, ny = yy.shape
        ns = TT.shape[0]

        At = np.zeros(shape=(nobs, ns))

        RQR = np.dot(np.dot(RR, QQ), RR.T)

        Pt = P0

        loglh = 0
        AA = At[0, :]

        for i in np.arange(0, nobs):

            At[i, :] = AA


            yhat = np.dot(ZZ, AA) + DD.flatten()
            nut = data[i, :] - yhat

            ind = (np.isnan(data[i, :])==False).nonzero()[0]
            nyact = ind.size

            Ft = np.dot(np.dot(ZZ[ind, :], Pt), ZZ[ind, :].T) + HH[ind, :][:, ind]
            Ft = 0.5*(Ft + Ft.T)

            dFt = np.log(np.linalg.det(Ft))

            iFtnut = sp.linalg.solve(Ft, nut[:, ind].T, sym_pos=True)

            if i+1 > t0:
                loglh = loglh - 0.5*nyact*np.log(2*np.pi) - 0.5*dFt \
                        - 0.5*np.dot(nut[:, ind], iFtnut)


            TTPt = np.dot(TT, Pt)

            Kt = np.dot(TTPt, ZZ[ind, :].T)

            AA = np.dot(TT, AA) + np.dot(Kt, iFtnut).squeeze()
            AA = np.asarray(AA).squeeze()

            Pt = np.dot(TTPt, TT.T) - np.dot(Kt, sp.linalg.solve(Ft, Kt.T, sym_pos=True)) + RQR



        results = {}
        results['filtered_states']   = p.DataFrame(At, columns=self.state_names)
        results['one_step_forecast'] = []
        results['log_lik'] = loglh[0, 0]


        return results



    def historical_decomposition(self, para, *args, **kwargs):
        pass



class LinLagExModel(StateSpaceModel):

    def __init__(self, yy, A, B, C, F, G, N, Q,
                 Aj, Bj, Cj, Fj, Gj,
                 Ainf, Binf, Cinf, Finf, Ginf,
                 t0=0,
                 shock_names=None, state_names=None, obs_names=None):
        import meyer_gohde
        self.A = A
        self.B = B
        self.C = C
        self.F = F
        self.G = G

        self.N = N
        self.Q = Q


        self.Aj = Aj
        self.Bj = Bj
        self.Cj = Cj
        self.Fj = Fj
        self.Gj = Gj

        self.Ainf = Ainf
        self.Binf = Binf
        self.Cinf = Cinf
        self.Finf = Finf
        self.Ginf = Ginf

        self.t0 = t0

        self.yy = yy

        self.shock_names = shock_names
        self.state_names = state_names
        self.obs_names = obs_names

    def find_max_it(self, p0):

        Aj = lambda j: np.array(self.Aj(p0, j), dtype=float)
        Bj = lambda j: np.array(self.Bj(p0, j), dtype=float)
        Cj = lambda j: np.array(self.Cj(p0, j), dtype=float)
        Fj = lambda j: np.array(self.Fj(p0, j), dtype=float)
        Gj = lambda j: np.array(self.Gj(p0, j), dtype=float)

        F = np.array(self.F(p0), dtype=float)

        find_max_it = meyer_gohde.mg.find_max_it
        max_it = find_max_it(Aj, Bj, Cj, Fj, Gj, F.shape[0], F.shape[1])

        return max_it

    def impulse_response(self, p0, h=20):

        A = np.array(self.A(p0), dtype=float)
        B = np.array(self.B(p0), dtype=float)
        C = np.array(self.C(p0), dtype=float)
        F = np.array(self.F(p0), dtype=float)
        G = np.array(self.G(p0), dtype=float)
        N = np.array(self.N(p0), dtype=float)
        Q = np.array(self.Q(p0), dtype=float)

        Aj = lambda j: np.array(self.Aj(p0, j), dtype=float)
        Bj = lambda j: np.array(self.Bj(p0, j), dtype=float)
        Cj = lambda j: np.array(self.Cj(p0, j), dtype=float)
        Fj = lambda j: np.array(self.Fj(p0, j), dtype=float)
        Gj = lambda j: np.array(self.Gj(p0, j), dtype=float)

        Ainf = np.array(self.Ainf(p0), dtype=float)
        Binf = np.array(self.Binf(p0), dtype=float)
        Cinf = np.array(self.Cinf(p0), dtype=float)
        Ginf = np.array(self.Ginf(p0), dtype=float)
        Finf = np.array(self.Finf(p0),dtype=float)


        ma_solve = meyer_gohde.mg.solve_ma_alt
        MA_VECTOR, ALPHA, BETA, RC = ma_solve(A, B, C, F, G, N,
                                              Aj, Bj, Cj, Fj, Gj,
                                              Ainf, Binf, Cinf, Ginf, Finf,h-1)

        nshocks = MA_VECTOR.shape[1]
        nvars = MA_VECTOR.shape[0]/h
        irfs = {}
        i = 0

        for respi in MA_VECTOR.T:
            irfs[self.shock_names[i]] = p.DataFrame(np.reshape(respi, (h, nvars))*np.sqrt(Q[i, i]), columns=self.state_names)
            i = i + 1
        return irfs


    def system_matrices(self, p0):
        pass


class LinearDSGEModel(StateSpaceModel):

    def __init__(self, yy, GAM0, GAM1, PSI, PPI,
                 QQ, DD, ZZ, HH, t0=0,
                 shock_names=None, state_names=None, obs_names=None):

        if len(yy.shape) < 2:
            yy = np.swapaxes(np.atleast_2d(yy), 0, 1)

        self.yy = yy

        self.GAM0 = GAM0
        self.GAM1 = GAM1
        self.PSI = PSI
        self.PPI = PPI
        self.QQ = QQ
        self.DD = DD
        self.ZZ = ZZ
        self.HH = HH

        self.t0 = t0

        self.shock_names = shock_names
        self.state_names = state_names
        self.obs_names = obs_names

    def solve_LRE(self, para, *args, **kwargs):

        G0 = self.GAM0(para, *args, **kwargs)
        G1 = self.GAM1(para, *args, **kwargs)
        PSI = self.PSI(para, *args, **kwargs)
        PPI = self.PPI(para, *args, **kwargs)
        C0 = np.zeros(G0.shape[0])

        TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensys.gensys_wrapper.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)

        return TT, RR, RC

    def system_matrices(self, para, *args, **kwargs):

        TT, RR, RC = self.solve_LRE(para, *args, **kwargs)

        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))
        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        return TT, RR, QQ, DD, ZZ, HH


if __name__ == '__main__':

    import numpy as np
    yy = np.random.rand(32)

    TT = lambda rho: rho
    RR = lambda rho: 1.0
    QQ = lambda rho: 1.0
    DD = lambda rho: 0.0
    ZZ = lambda rho: 1.0
    HH = lambda rho: 0.0

    test_ss = StateSpaceModel(yy, TT, RR, QQ, DD, ZZ, HH)
    print test_ss.system_matrices(0.3)
    print test_ss.log_lik(0.3)
