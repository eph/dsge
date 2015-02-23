from __future__ import division

import numpy as np
import scipy as sp
import pandas as p
from fortran import dlyap, gensysw, filter
from helper_functions import cholpsd


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
        self.state_names = None
        self.obs_names = None

    def log_lik(self, para, *args, **kwargs):

        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0, info = dlyap.dlyap(TT, RR.dot(QQ).dot(RR.T))

        lik = filter.filter.kalman_filter(yy.T, TT, RR, QQ, DD, ZZ, HH, P0)

        return lik

    def log_quasilik_hstep(self, para, h=4, *args, **kwargs):

        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0, info = dlyap.dlyap(TT, RR.dot(QQ).dot(RR.T))

        lik = filter.filter.kalman_filter_hstep_quasilik(yy.T, TT, RR, QQ, DD, ZZ, HH, P0, h)

        return lik


    def kf_everything(self, para, *args, **kwargs):
        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        f = filter.filter.kalman_filter_missing_with_states
        loglh, filtered_states, smoothed_states = f(yy.T, TT, RR, QQ, DD.squeeze(), ZZ, HH, t0)

        results = {}
        results['log_lik'] = p.DataFrame(loglh, columns=['log_lik'])

        results['log_lik'].index = yy.index
        filtered_states = p.DataFrame(filtered_states, columns=self.state_names)
        filtered_states.index = yy.index
        results['filtered_states'] = filtered_states

        smoothed_states = p.DataFrame(smoothed_states, columns=self.state_names)
        smoothed_states.index = yy.index
        results['smoothed_states'] = smoothed_states

        return results

    def pred(self, para, h=20, shocks=True, append=False, *args, **kwargs):

        yy = kwargs.pop('y', self.yy)
        res = self.kf_everything(para, y=yy, *args, **kwargs)

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)



        At = res['smoothed_states'].iloc[-1].values
        ysim  = np.zeros((h, DD.size))

        index0 = res['smoothed_states'].index[-1]+1
        index = []
        for i in range(h):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = TT.dot(At) + shocks*RR.dot(e)

            h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
            At = np.asarray(At).squeeze()
            ysim[i, :] = DD.T + ZZ.dot(At) + shocks*np.atleast_2d(h)
            index.append(index0+i)

        ysim = p.DataFrame(ysim, columns=self.obs_names, index=index)

        if append:
            ysim = self.yy.append(ysim)
        return ysim

    def log_lik_pff(self, para, *args, **kwargs):

        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')
        npart = kwargs.pop('npart', 1000)
        filt = kwargs.pop('filt', 'bootstrap')
        seed = kwargs.pop('seed', 0)
        resampling = kwargs.pop('resampling', 'stratified')
        reduce_system = kwargs.pop('reduce_system', False)

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if reduce_system:
            f = filter.filter.kalman_filter_missing_with_states
            loglh, filtered_states, smoothed_states = f(yy.T, TT, RR, QQ, DD, ZZ, HH, t0)

            from scipy.linalg import schur
            (Z, T, nmin) = schur(TT, sort=lambda x: abs(x**2)>1e-15)

            RRhat = Z.T.dot(RR)
            ishock = np.argwhere(abs(RRhat[nmin:, :]).sum(0)>1e-10).flatten()
            neshock = ishock.size

            if neshock>0:
                TThat = np.hstack((T[:nmin,:][:, :nmin], T[:nmin, :][:, nmin:].dot(RRhat[nmin:, :][:, ishock])))
                TThat = np.vstack((TThat, np.zeros((neshock, nmin+neshock))))

                imat = np.zeros((TT.shape[0], neshock+nmin))
                imat[:nmin, :][:, :nmin] = np.eye(nmin)
                imat[nmin:, :][:, nmin:] = RRhat[nmin:, :][:, ishock]

                x = np.zeros((neshock, RR.shape[1]))
                x[:, ishock] = np.eye(neshock)

                RRhat = np.vstack((RRhat[:nmin, :], x))
                ZZhat = ZZ.dot(Z).dot(imat)

            else:
                jfdsklf

            loglh, filtered_states, smoothed_states = f(yy.T, TT, RR, QQ, DD, ZZ, HH, t0)

            TT = TThat.copy()
            RR = RRhat.copy()
            ZZ = ZZhat.copy()


        filti = {'bootstrap':0, 'cond-opt': 1}
        fi = filti[filt]

        resamp = {'systematic':0, 'multinomial': 1, 'stratified': 2}
        ri = resamp[resampling]

        lik, filtered_states = filter.filter.part_filter(yy.T, TT, RR, QQ, DD, ZZ, HH, t0, npart, fi, ri, seed)
        results = {}
        results['log_lik'] = p.DataFrame(lik, columns=['log_lik'])
        results['log_lik'].index = yy.index
        filtered_states = p.DataFrame(filtered_states, columns=self.state_names)
        filtered_states.index = yy.index
        results['filtered_states'] = filtered_states

        return results


    def log_lik_pf(self, para, *args, **kwargs):

        t0 = kwargs.pop('t0', self.t0)
        npart = kwargs.pop('npart', 1000)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')
        filt = kwargs.pop('filt', 'bootstrap')
        rcond = kwargs.pop('rcond',1e-7)


        data = np.asarray(yy)
        data = np.atleast_2d(data)
        seed = kwargs.pop('seed', None)
        scale = kwargs.pop('scale', 10)
        init_s_w = kwargs.pop('init_s_w', None)

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0, info = dlyap.dlyap(TT, RR.dot(QQ).dot(RR.T))

        if seed is not None:
            np.random.seed(seed)

        from scipy.stats import multivariate_normal

        RQR = RR.dot(QQ).dot(RR.T)
        P0 = TT.dot(P0).dot(TT.T) + RQR
        nobs = data.shape[0]
        ny = data.shape[1]
        neps = QQ.shape[0]
        ns = TT.shape[0]

        St = np.zeros((nobs+1, ns, npart))

        U,s,V = sp.linalg.svd(P0)
        L = U.dot(np.diag(np.sqrt(s)))
        St[0,:,:] = L.dot(np.random.normal(size=(ns,npart)))



        RRcQQ = RR.dot(np.linalg.cholesky(QQ))
        iRQR = np.linalg.pinv(RQR,rcond=1e-7)
        U,S,V = np.linalg.svd(RQR,full_matrices=0)
        detRQR = np.sum(np.log(S[S>1e-7]))

        ps = multivariate_normal(cov=RQR)

        # container variables
        resamp = np.zeros((nobs))
        wtsim = np.ones((nobs+1,npart))
        incwt = np.zeros((nobs,npart))
        incwtaux = np.zeros((nobs,npart))

        #------------------------------------------------------------
        # p(yt|st) [in terms of dev from mean]
        #------------------------------------------------------------
        py = multivariate_normal(cov=HH)
        ESS = np.zeros((nobs))
        loglh = np.zeros((nobs))
        loglhalt = np.zeros((nobs))




        #------------------------------------------------------------
        # Helper Functions
        #------------------------------------------------------------
        demeaned_data = data - np.tile(np.squeeze(DD), (nobs, 1))
        fcst_error = lambda i: np.tile(demeaned_data[i].T, (1, npart))-ZZ.dot(TT.dot(St[i, ...]))
        if filt=='cond-opt':
            Pt = RQR
            Ft = np.dot(np.dot(ZZ, Pt), ZZ.T) + HH
            iFt = np.linalg.inv(Ft)
            PtZZpiFt = Pt.dot(ZZ.T).dot(iFt)
            Pt = Pt - Pt.dot(ZZ.T).dot(iFt).dot(ZZ).dot(Pt)
            pys = multivariate_normal(cov=Pt)


        #------------------------------------------------------------
        # AUXILIARY PARTICLE FILTER
        #------------------------------------------------------------
        if filt == 'auxsimp':
            ptilde = multivariate_normal(cov=scale*HH)
            fcst_error = lambda i: np.tile(demeaned_data[i].T, (1, npart))-ZZ.dot(TT.dot(St[i, ...]))

            phatold = ptilde.logpdf(fcst_error(0).T)

            wtsim[0, :] = np.exp(phatold)
            wtsim[0, :] = wtsim[0, :]/(wtsim[0, :].mean())

        if init_s_w is not None:
            St[0, :, :] = init_s_w[0]
            wtsim[0, :] = init_s_w[1]

        for t in range(nobs):
            if filt=='bootstrap':
                #eps = RRcQQ.dot(np.random.normal(size=(neps, npart)))
                St[t+1,...] = TT.dot(St[t,...]) + ps.rvs(size=npart).T#eps
            elif filt=='cond-opt':
                nut = fcst_error(t)
                Stbar = TT.dot(St[t,...]) + PtZZpiFt.dot(nut)
                St[t+1, ...] = Stbar + pys.rvs(size=npart).T
                lng = pys.logpdf( (St[t+1, ...]-Stbar).T )

            elif filt=='auxsimp':
                #------------------------------------------------------------
                # Simulate
                #------------------------------------------------------------
                St[t+1,...] = TT.dot(St[t,...]) + ps.rvs(size=npart).T

                phatold = ptilde.logpdf(fcst_error(t).T)
                if t < nobs-1:
                    phat = ptilde.logpdf(fcst_error(t+1).T)



            nut = (np.tile(demeaned_data[t].T, (1, npart))
                           - ZZ.dot(St[t+1,...]))

            incwt[t,:] = py.logpdf(nut.T)

            if filt=='cond-opt':
                eta = St[t+1,...] - TT.dot(St[t,...])
                lnps = (-0.5*neps*np.log(2.0*np.pi) - 0.5*detRQR
                        -0.5*np.einsum('ji,ji->i', np.dot(iRQR,eta),eta))
                incwt[t,:] = incwt[t,:] + lnps - lng

            elif filt=='auxsimp':
                incwtaux = incwt[t,:] - phatold
                incwt[t, :] = incwt[t,:] + phat - phatold

            # Equation 7.7
            wtsim[t+1,:] = np.exp(incwt[t,:]) * wtsim[t,:]
            wtsim[t+1,:] = wtsim[t+1,:]/np.mean(wtsim[t+1,:])

            ESS[t] = npart / np.mean(wtsim[t+1,:]**2)
            loglh[t] = np.log(np.mean(np.exp(incwt[t,:])*wtsim[t,:]))

            if filt=='auxsimp':
                if t==0:
                    loglh[t] = (np.log(np.mean(np.exp(phatold))) +
                                np.log(np.mean(np.exp(incwtaux)*wtsim[t,:])))
                    loglhalt[t] = np.log(np.mean(np.exp(incwt[t, :])*wtsim[t, :])) + np.log(np.mean(np.exp(phatold)))
                elif t==nobs-1:
                    loglh[t] = (-np.log(np.mean((1/np.exp(phatold))*wtsim[t, :])) +
                                np.log(np.mean(np.exp(incwtaux)*wtsim[t,:])) )

                    loglhalt[t] = np.log(np.mean(np.exp(incwtaux)*wtsim[t, :]))
                else:
                    loglh[t] = (-np.log(np.mean((1/np.exp(phatold))*wtsim[t, :])) +
                                np.log(np.mean(np.exp(incwtaux)*wtsim[t,:])) )
                    loglhalt[t] = np.log(np.mean(np.exp(incwt[t, :])*wtsim[t, :]))

                    #loglhalt[t] =
            if ESS[t] < npart/2:
                from fortran import filter
                resamp = filter.filter.sys_resampling
                ind = resamp(wtsim[t+1,:]/sum(wtsim[t+1,:]), np.random.rand())
                if ns == 1:
                    St[t+1,...] = np.squeeze(St[t+1,:,ind-1])
                else:
                    St[t+1,...] = St[t+1,...][:,ind-1]

                wtsim[t+1,:] = 1
                wtsim[t, :] = wtsim[t, ind-1]
                incwt[t, :] = incwt[t, ind-1]
                if t > 0:
                    wtsim[t-1, :] = wtsim[t-1, ind-1]
                    incwt[t-1, :] = incwt[t-1, ind-1]

        filtered_states = TT.dot(St[:-1,...,...]) * wtsim[:-1,:]
        filtered_states = filtered_states.mean(2).T

        if isinstance(yy,p.DataFrame):
            loglh = p.DataFrame(loglh,columns=['Log Lik.'])
            loglh.index = yy.index

            ESS = p.DataFrame(ESS,columns=['ESS'])
            ESS.index = yy.index

            filtered_states = p.DataFrame(filtered_states, columns=self.state_names)
            filtered_states.index = yy.index

        results = {}
        results['log_lik'] = loglh
        results['log_likalt'] = loglhalt
        results['ESS'] = ESS
        results['filtered_states'] = filtered_states
        results['St'] = St[t+1, ...]
        results['wt'] = wtsim[t+1, ...]
        results['incwt'] = incwt
        return results

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

    def simulate(self, para, nsim=200, *args, **kwargs):

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)
        ysim  = np.zeros((nsim*2, DD.size))
        At = np.zeros((TT.shape[0],))

        for i in range(nsim*2):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = TT.dot(At) + RR.dot(e)

            h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
            At = np.asarray(At).squeeze()
            ysim[i, :] = DD.T + ZZ.dot(At) + np.atleast_2d(h)

        return ysim[nsim:, :]


    def forecast(self, para, h=20, shocks=True, *args, **kwargs):
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

        ZZ = np.atleast_2d(ZZ)
        HH = np.atleast_2d(HH)
        loglh = np.zeros((nobs))
        AA = At[0, :]

        for i in np.arange(0, nobs):

            At[i, :] = AA


            yhat = np.dot(ZZ, AA) + DD.flatten()
            nut = data[i, :] - yhat
            nut = np.atleast_2d(nut)
            ind = (np.isnan(data[i, :])==False).nonzero()[0]
            nyact = ind.size

            Ft = np.dot(np.dot(ZZ[ind, :], Pt), ZZ[ind, :].T) + HH[ind, :][:, ind]
            Ft = 0.5*(Ft + Ft.T)

            dFt = np.log(np.linalg.det(Ft))

            iFtnut = sp.linalg.solve(Ft, nut[:, ind].T, sym_pos=True)

            if i+1 > t0:
                loglh[i]= - 0.5*nyact*np.log(2*np.pi) - 0.5*dFt \
                        - 0.5*np.dot(nut[:, ind], iFtnut)


            TTPt = np.dot(TT, Pt)

            Kt = np.dot(TTPt, ZZ[ind, :].T)

            AA = np.dot(TT, AA) + np.dot(Kt, iFtnut).squeeze()
            AA = np.asarray(AA).squeeze()

            Pt = np.dot(TTPt, TT.T) - np.dot(Kt, sp.linalg.solve(Ft, Kt.T, sym_pos=True)) + RQR


        if isinstance(yy,p.DataFrame):
            loglh = p.DataFrame(loglh,columns=['Log Lik.'])
            loglh.index = yy.index


        results = {}
        results['filtered_states']   = p.DataFrame(At, columns=self.state_names,index=yy.index)
        results['one_step_forecast'] = []
        results['log_lik'] = loglh

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
                 shock_names=None, state_names=None, obs_names=None,
                 prior=None):

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

        self.prior = prior

    def solve_LRE(self, para, *args, **kwargs):

        G0 = self.GAM0(para, *args, **kwargs)
        G1 = self.GAM1(para, *args, **kwargs)
        PSI = self.PSI(para, *args, **kwargs)
        PPI = self.PPI(para, *args, **kwargs)
        C0 = np.zeros(G0.shape[0])

        nf = PPI.shape[1]

        if nf > 0:
            TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensysw.gensys_wrapper.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)
        else:
            TT = np.linalg.inv(G0).dot(G1)
            RR = np.linalg.inv(G0).dot(PSI)
            RC = 1
        return TT, RR, RC

    def system_matrices(self, para, *args, **kwargs):

        TT, RR, RC = self.solve_LRE(para, *args, **kwargs)

        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))
        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        return TT, RR, QQ, DD, ZZ, HH

    def log_pr(self, para, *args, **kwargs):
        try:
            return self.prior.logpdf(para)
        except:
            pass
            #raise("no prior specified")
    def log_post(self, para, *args, **kwargs):
        x = self.log_lik(para) + self.log_pr(para)
        if np.isnan(x):
            x = -1000000000
        return x



if __name__ == '__main__':

    import numpy as np
    yy = np.random.rand(32)

    TT = lambda rho: rho
    RR = lambda rho: 1.0
    QQ = lambda rho: 1.0
    DD = lambda rho: 0.0
    ZZ = lambda rho: 1.0
    HH = lambda rho: 0.1

    test_ss = StateSpaceModel(yy, TT, RR, QQ, DD, ZZ, HH)
    print test_ss.system_matrices(0.3)
    print test_ss.log_lik(0.3)
