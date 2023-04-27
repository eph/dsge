"""
A module for Linear Gaussian State Space models.


Classes
-------
StateSpaceModel
LinearDSGEModel
"""
import numpy as np
import pandas as p

from scipy.linalg import solve_discrete_lyapunov

from .gensys import gensys
from .filters import chand_recursion, kalman_filter, filter_and_smooth

filt_choices = {"chand_recursion": chand_recursion,
                "kalman_filter": kalman_filter}


class StateSpaceModel(object):
    r"""
    Object for holding state space model

    .. math::

    s_t &=& T(\theta) s_{t-1} + R(\theta) \epsilon_t,
    \quad \epsilon_t \sim N(0,Q(\theta)) \\

    y_t &=& D(\theta) + Z(\theta) s_{t} + \eta_t,
    \quad \eta_t  \sim N(0, H(\theta))

    Attributes
    ----------
    yy : array_like or pandas.DataFrame
        Dataset containing the observables of the model.

    t0 : int, optional
        Number of initial observations to condition on for
        likelihood evaluation. The default is 0.

    shock_names : list or None, optional
        Names of the shocks.

    state_names : list or None, optional
        Names of the states.

    obs_names : list or None, optional
        Names of observables.

    Methods
    -------
    TT(para), RR(para), QQ(para)
        Define the state transition matrices as function of a parameter vector.
    DD(para), ZZ(para), HH(para)
        Define the observable transition matrices as function of a parameter vector.
    log_lik(para)
        Computes the likelihood of the model at parameter value para.
    impulse_response(para, h=20)
        Computes the impulse response function at parameter value para.
    pred(para, h=20, shocks=True, append=False)
        Simulates a draw from the predictive distribution at parameter
        value para.
    kf_everything(para)
        Generates the filtered and smoothed posterior means of the state vector.
    """

    fast_filter = "chand_recursion"

    def __init__(
        self,
        yy,
        CC,
        TT,
        RR,
        QQ,
        DD,
        ZZ,
        HH,
        t0=0,
        shock_names=None,
        state_names=None,
        obs_names=None,
    ):

        if len(yy.shape) < 2:
            yy = np.swapaxes(np.atleast_2d(yy), 0, 1)

        self.yy = yy

        self.CC = CC
        self.TT = TT
        self.RR = RR
        self.QQ = QQ
        self.DD = DD
        self.ZZ = ZZ
        self.HH = HH

        self.t0 = t0

        self.shock_names = shock_names
        self.state_names = state_names
        self.obs_names = obs_names

    def log_lik(self, para, *args, **kwargs):
        """
        Computes the log likelihood of the model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        t0 : int, optional
            Number of initial observations to condition on. 
        y : 2d array-like, optional
            Dataset of observables (T x nobs). The default is the observable set pass during
            class instantiation.
        P0 : 2d array-like or string, optional
            [ns x ns] initial covariance matrix of states, or `unconditional` to use the one
            associated with the invariant distribution.  The default is `unconditional.`


        Returns
        -------
        lik : float
            The log likelihood.


        See Also
        --------
        StateSpaceModel.kf_everything

        """

        t0 = kwargs.pop("t0", self.t0)
        yy = kwargs.pop("y", self.yy)
        P0 = kwargs.pop("P0", "unconditional")

        if np.isnan(yy).any().any():
            default_filter = "kalman_filter"
        else:
            default_filter = "chand_recursion"

        filt = kwargs.pop("filter", default_filter)
        filt_func = filt_choices[filt]

        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para)
        A0 = kwargs.pop("A0", np.zeros(CC.shape))
        if (np.isnan(TT)).any():
            lik = -1000000000000.0
            return lik

        if P0 == "unconditional":
            P0 = solve_discrete_lyapunov(TT, RR.dot(QQ).dot(RR.T))

        lik = filt_func(
            np.asarray(yy),
            CC,
            TT,
            RR,
            QQ,
            np.asarray(DD, dtype=float),
            np.asarray(ZZ, dtype=float),
            np.asarray(HH, dtype=float),
            np.asarray(A0, dtype=float),
            np.asarray(P0, dtype=float),
            t0=t0,
        )
        return lik

    def kf_everything(self, para, *args, **kwargs):
        """
        Runs the kalman filter and returns objects of interest.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        t0 : int, optional
            Number of initial observations to condition on. 
        y : 2d array-like, optional
            Dataset of observables (T x nobs). The default is the observable set pass during
            class instantiation.
        P0 : 2d arry-like or string, optional
            [ns x ns] initial covariance matrix of states, or `unconditional` to use the one
            associated with the invariant distribution.  The default is `unconditional.`
        shocks : bool, optional
            Whether to filter and smooth for the structural shocks as well as states. 
            The default is True. 

        Returns
        -------
        results : dict of p.DataFrames with
             `log_lik` -- the sequence of log likelihoods
             `filtered_means' -- the filtered means of the states 
             `filtered_std' -- the filtered stds of the states
             `forecast_means' -- the forecasted means of the states 
             `forecast_std' -- the forecasted stds of the states
             `smoothed_means' -- the smoothed means of the model
             `smoothed_stds' -- the smoothed stds of the model 

        Notes
        -----
        Can be used with missing (NaN) observations.
        """

        t0 = kwargs.pop("t0", self.t0)
        yy = kwargs.pop("y", self.yy)
        P0 = kwargs.pop("P0", "unconditional")
        get_shocks = kwargs.pop("shocks", True)
        yy = p.DataFrame(yy)

        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)
        if get_shocks:
            from scipy.linalg import block_diag

            neps = RR.shape[1]
            nobs = ZZ.shape[0]
            TT = block_diag(TT, np.zeros((neps, neps)))
            RR = np.vstack([RR, np.eye(neps)])
            CC = np.zeros((TT.shape[0]))
            ZZ = np.hstack([ZZ, np.zeros((nobs, neps))])

        A0 = kwargs.pop("A0", np.zeros(CC.shape))
        if P0 == "unconditional":
            P0 = solve_discrete_lyapunov(TT, RR.dot(QQ).dot(RR.T))

        res = filter_and_smooth(
            np.asarray(yy),
            CC,
            TT,
            RR,
            QQ,
            np.asarray(DD, dtype=float),
            np.asarray(ZZ, dtype=float),
            np.asarray(HH, dtype=float),
            np.asarray(A0, dtype=float),
            np.asarray(P0, dtype=float),
            t0=t0,
        )

        (
            loglh,
            filtered_means,
            filtered_stds,
            filtered_cov,
            forecast_means,
            forecast_stds,
            forecast_cov,
            smoothed_means,
            smoothed_stds,
            smoothed_cov,
        ) = res

        results = {}
        results["log_lik"] = p.DataFrame(loglh, columns=["log_lik"], index=yy.index)

        for resname, res in [
            ("filtered_means", filtered_means),
            ("filtered_stds", filtered_stds),
            ("forecast_means", forecast_means),
            ("forecast_stds", forecast_stds),
            ("smoothed_means", smoothed_means),
            ("smoothed_stds", smoothed_stds),
        ]:

            if get_shocks:
                names = self.state_names + self.shock_names
            else:
                names = self.state_names
            resdf = p.DataFrame(res, columns=names, index=yy.index)
            results[resname] = resdf

        results['forecast_cov'] = forecast_cov
        results['filtered_cov'] = filtered_cov
        results['smoothed_cov'] = smoothed_cov
        return results

    def pred(self, para, h=20, shocks=True, append=False, return_states=False, filt_para=None, *args, **kwargs):
        """
        Draws from the predictive distribution $p(Y_{t+1:t+h}|Y_{1:T}, \theta)$.


        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        h : int, optional
            The horizon of the distribution.
        y : 2d array-like, optional
            Dataset of observables (T x nobs). The default is the observable set pass during
            class instantiation.
        append : bool, optional
            Return the draw appended to yy (default = FALSE).

        Returns
        -------
        ysim : pandas.DataFrame
            A dataframe containing the draw from the predictive distribution.

        """
        if filt_para is None:
            filt_para = para

        yy = kwargs.pop("y", self.yy)
        res = self.kf_everything(filt_para, y=yy, shocks=False, *args, **kwargs)

        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        At = res["smoothed_means"].iloc[-1].values
        ysim = np.zeros((h, DD.size))

        asim = np.zeros((h, At.size))
        index0 = res["smoothed_means"].index[-1] + 1
        index = []
        for i in range(h):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = CC + TT.dot(At) + shocks * RR.dot(e)
            h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
            At = np.asarray(At).squeeze()
            ysim[i, :] = DD.T + ZZ.dot(At) + shocks * np.atleast_2d(h)
            asim[i, :] = At.copy()
            index.append(index0 + i)

        ysim = p.DataFrame(ysim, columns=self.obs_names, index=index)
        asim = p.DataFrame(asim, columns=self.state_names, index=index)
        if append:
            yhat = yy.copy()
            ysim = yhat.append(ysim)
            r = res['smoothed_means'].copy()
            asim = r.append(asim)

        if return_states:
            return ysim, asim
        else:
            return ysim

    def system_matrices(self, para, *args, **kwargs):
        """
        Returns the system matrices of the state space model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.


        Returns
        -------
        CC : np.array (ns)
        TT : np.array (ns x ns)
        RR : np.array (ns x neps)
        QQ : np.array (neps x neps)
        DD : np.array (nobs)
        ZZ : np.array (ny x ns)
        HH : np.array (ny x ny)


        Notes
        -----
        
        """
        CC = np.atleast_1d(self.CC(para, *args, **kwargs))
        TT = np.atleast_2d(self.TT(para, *args, **kwargs))
        RR = np.atleast_2d(self.RR(para, *args, **kwargs))
        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))

        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        return CC, TT, RR, QQ, DD, ZZ, HH

    def abcd_representation(self, para, *args, **kwargs):
        """
        Returns ABCD representation of state space system.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        

        Returns 
        -------
        A, B, C, D


        Notes
        -----
        

        """

        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        A = TT
        B = RR

        C = ZZ.dot(TT)  # ZZ @ TT
        D = ZZ.dot(RR)

        return A, B, C, D

    def impulse_response(self, para, h=20, *args, **kwargs):
        """
        Computes impulse response functions of model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.

        h : int, optional
           The maximum horizon length for the impulse responses.


        Returns
        -------
        irf : pandas.Panel (nshocks x h x nvariables)


        Notes
        -----
        These are of the model state variables impulse responses to
        1 standard deviation shocks.  The method does not return IRFs in terms
        of the model observables.
        """
        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if self.shock_names == None:
            self.shock_names = ["shock_" + str(i) for i in range(QQ.shape[0])]
            self.state_names = ["state_" + str(i) for i in range(TT.shape[0])]
            self.obs_names = ["obs_" + str(i) for i in range(ZZ.shape[0])]

        neps = QQ.shape[0]

        irfs = {}
        for i in range(neps):

            At = np.zeros((TT.shape[0], h + 1))
            QQz = np.zeros_like(QQ)
            QQz[i, i] = QQ[i, i]
            cQQz = np.sqrt(QQz)

            # cQQz = np.linalg.cholesky(QQz)

            At[:, 0] = (RR.dot(cQQz)[:, i]).squeeze()

            for j in range(h):
                At[:, j + 1] = TT.dot(At[:, j])

            irfs[self.shock_names[i]] = p.DataFrame(At.T, columns=self.state_names)

        return irfs

    def simulate(self, para, nsim=200, *args, **kwargs):
        """
        Simulates the observables of the model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.

        nsim : int, optional
            The length of the simulation. The default value is 200.

        Returns
        -------
        ysim : np.array (nsim x nobs)


        Notes
        -----
        The simulation is initialized from the steady-state mean and subsequently
        a simulation of length 2*nsim is created, with the final nsim observations
        return.
        """

        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)
        ysim = np.zeros((nsim * 2, DD.size))
        At = np.zeros((TT.shape[0],))

        for i in range(nsim * 2):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = CC + TT.dot(At) + RR.dot(e)

            h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
            At = np.asarray(At).squeeze()
            ysim[i, :] = DD.T + ZZ.dot(At) + np.atleast_2d(h)

        return ysim[nsim:, :]

    def historical_decomposition(self, p0, init=1):
        shock_decomposition = []
     
        CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(p0)
        states = self.kf_everything(p0, shocks=True)
         
        shocks = states['smoothed_means'][self.shock_names]
        nshocks = len(self.shock_names)
        shocks = shocks.values
        dfs = []
         
        from scipy.linalg import block_diag
        TT = block_diag(TT,np.zeros((nshocks, nshocks)))
        RR = np.r_[RR, np.eye(nshocks)]
        for i, shock in enumerate(self.shock_names):
            T, ns = states['smoothed_means'].shape
         
            decomp = np.zeros_like(states['smoothed_means'].values)
            for j in range(init,T):
                decomp[j] = TT @ decomp[j-1] + RR[:,i] * shocks[j,i]
         
         
            dfs.append(p.DataFrame(decomp, 
                       index=self.yy.index, 
                       columns=self.state_names+self.shock_names))
         
         
         
        decomp = np.zeros_like(states['smoothed_means'].values)
        decomp[init-1] = states['smoothed_means'].values[init-1]
        for j in range(init,T):
            decomp[j] = TT @ decomp[j-1] 
         
        dfs.append(p.DataFrame(decomp, 
                   index=self.yy.index, 
                   columns=self.state_names+self.shock_names))
         
        shock_decomposition = p.concat(dfs, keys=self.shock_names+['Initial Condition'])
         
         
        return shock_decomposition


class LinearDSGEModel(StateSpaceModel):
    def __init__(
        self,
        yy,
        GAM0,
        GAM1,
        PSI,
        PPI,
        QQ,
        DD,
        ZZ,
        HH,
        t0=0,
        shock_names=None,
        state_names=None,
        obs_names=None,
        prior=None,
        parameter_names=None
    ):

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
        self.parameter_names = parameter_names

        self.prior = prior

    def solve_LRE(self, para, *args, **kwargs):

        G0 = self.GAM0(para, *args, **kwargs)
        G1 = self.GAM1(para, *args, **kwargs)
        PSI = self.PSI(para, *args, **kwargs)
        PPI = self.PPI(para, *args, **kwargs)

        G0 = np.atleast_2d(G0)
        G1 = np.atleast_2d(G1)
        PSI = np.atleast_2d(PSI)
        PPI = np.atleast_2d(PPI)

        C0 = np.zeros(G0.shape[0])

        nf = PPI.shape[1]

        if nf > 0:
            TT, RR, RC = gensys(G0, G1, PSI, PPI, C0)
            RC = RC[0] * RC[1]
            # TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensysw.gensys.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)
        else:
            TT = np.linalg.inv(G0).dot(G1)
            RR = np.linalg.inv(G0).dot(PSI)
            RC = 1

        return TT, RR, RC

    def system_matrices(self, para, *args, **kwargs):

        TT, RR, RC = self.solve_LRE(para, *args, **kwargs)
        CC = np.zeros(TT.shape[0])

        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))
        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        if RC != 1:
            TT = np.nan * TT

        return CC, TT, RR, QQ, DD, ZZ, HH

    def log_pr(self, para, *args, **kwargs):
        try:
            return self.prior.logpdf(para)
        except:
            pass
            # raise("no prior specified")

    def log_post(self, para, *args, **kwargs):
        x = self.log_lik(para) + self.log_pr(para)
        if np.isnan(x):
            x = -1000000000.0
        if x < -1000000000.0:
            x = -1000000000.0
        return x
