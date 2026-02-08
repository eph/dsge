"""
Linear Gaussian State Space models and a DSGE-specialized subclass.

This module provides:
- StateSpaceModel: a light wrapper around linear Gaussian state-space models,
  with likelihood, filtering/smoothing, impulse responses, simulation, and
  helper utilities.
- LinearDSGEModel: a subclass that binds the canonical DSGE linear form
  (GAM0/GAM1/PSI/PPI) and solves it via `gensys` to produce state-space
  matrices consumable by `StateSpaceModel` APIs.
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
    Linear Gaussian state‑space model

    The model is specified by the following system (parameterized by θ):

    .. math::

        s_t &= C(𝜃) + T(𝜃) s_{t-1} + R(𝜃)\, \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0, Q(𝜃)) \\
        y_t &= D(𝜃) + Z(𝜃) s_t + \eta_t,\quad\ \eta_t \sim \mathcal{N}(0, H(𝜃))

    This class expects callables `CC, TT, RR, QQ, DD, ZZ, HH` that map a
    parameter vector to the corresponding arrays (with shapes documented
    below). Most users will work through `LinearDSGEModel`, which constructs
    these for you from a linearized DSGE system.

    Attributes
    ----------
    yy : array_like or pandas.DataFrame
        Dataset of observables (shape T x nobs). A 1D input is promoted to 2D.
    t0 : int
        Number of initial observations to condition on for likelihood.
    shock_names, state_names, obs_names : list[str] or None
        Optional labels for shocks, states, and observables.

    Notes
    -----
    - `fast_filter` selects the default filter for complete data
      ("chand_recursion"). If missing values are present, Kalman filter is used.
    - Set `use_cache=True` to reuse matrices from the most recent call to
      `system_matrices`, avoiding recomputation when scanning nearby params.
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

        self.cached_system_matrices = None

    def log_lik(self, para, use_cache=False, *args, **kwargs):
        """
        Compute the Gaussian log-likelihood at parameters `para`.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        t0 : int, optional
            Number of initial observations to condition on. 
        y : 2d array-like, optional
            Dataset of observables (T x nobs). Defaults to the dataset passed
            during class instantiation.
        P0 : 2d array-like or string, optional
            [ns x ns] initial covariance matrix of states, or `unconditional` to use the one
            associated with the invariant distribution.  The default is `unconditional.`


        Returns
        -------
        lik : float
            The log-likelihood value.


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

        if use_cache:
            CC, TT, RR, QQ, DD, ZZ, HH = self.cached_system_matrices
        else:
            CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para)

        A0 = kwargs.pop("A0", np.zeros(CC.shape))
        if (np.isnan(TT)).any():
            lik = -1000000000000.0
            return lik

        if P0 == "unconditional":
            P0 = solve_discrete_lyapunov(TT, RR.dot(QQ).dot(RR.T))

        lik = filt_func(
            np.ascontiguousarray(yy),
            np.ascontiguousarray(CC),
            np.ascontiguousarray(TT),
            np.ascontiguousarray(RR),
            np.ascontiguousarray(QQ),
            np.ascontiguousarray(DD, dtype=float),
            np.ascontiguousarray(ZZ, dtype=float),
            np.ascontiguousarray(HH, dtype=float),
            np.ascontiguousarray(A0, dtype=float),
            np.ascontiguousarray(P0, dtype=float),
            t0=t0,
        )
        return lik

    def kf_everything(self, para, use_cache=False, *args, **kwargs):
        """
        Run Kalman filtering and smoothing, return common outputs.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        t0 : int, optional
            Number of initial observations to condition on. 
        y : 2d array-like, optional
            Dataset of observables (T x nobs). Defaults to the dataset passed
            during class instantiation.
        P0 : 2d arry-like or string, optional
            [ns x ns] initial covariance matrix of states, or `unconditional` to use the one
            associated with the invariant distribution.  The default is `unconditional.`
        shocks : bool, optional
            Whether to filter and smooth for the structural shocks as well as states. 
            The default is True. 

        Returns
        -------
        results : dict of p.DataFrames
            Keys include `log_lik`, `filtered_means`, `filtered_stds`,
            `forecast_means`, `forecast_stds`, `smoothed_means`, `smoothed_stds`.
            Raw covariances are also included under `filtered_cov`, `forecast_cov`,
            and `smoothed_cov`.

        Notes
        -----
        Can be used with missing (NaN) observations.
        """

        t0 = kwargs.pop("t0", self.t0)
        yy = kwargs.pop("y", self.yy)
        P0 = kwargs.pop("P0", "unconditional")
        get_shocks = kwargs.pop("shocks", True)
        yy = p.DataFrame(yy)

        if use_cache:
            CC, TT, RR, QQ, DD, ZZ, HH = self.cached_system_matrices
        else:
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
            np.ascontiguousarray(CC),
            np.ascontiguousarray(TT),
            np.ascontiguousarray(RR),
            np.ascontiguousarray(QQ),
            np.ascontiguousarray(DD, dtype=float),
            np.ascontiguousarray(ZZ, dtype=float),
            np.ascontiguousarray(HH, dtype=float),
            np.ascontiguousarray(A0, dtype=float),
            np.ascontiguousarray(P0, dtype=float),
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

    def pred(self, para, h=20, shocks=True, append=False, return_states=False, filt_para=None, use_cache=False, *args, **kwargs):
        """
        Draw from the predictive distribution :math:`p(Y_{t+1:t+h}|Y_{1:T}, \theta)`.


        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        h : int, optional
            The horizon of the distribution.
        y : 2d array-like, optional
            Dataset of observables (T x nobs). Defaults to the dataset passed
            during class instantiation.
        append : bool, optional
            Return the draw appended to yy (default = FALSE).

        Returns
        -------
        ysim : pandas.DataFrame (and optionally states)
            Simulated observables (and optionally states if `return_states=True`).

        """
        if filt_para is None:
            filt_para = para

        yy = kwargs.pop("y", self.yy)
        res = self.kf_everything(filt_para, y=yy, shocks=False, *args, **kwargs)

        if use_cache:
            CC, TT, RR, QQ, DD, ZZ, HH = self.cached_system_matrices
        else:
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
            yhat = p.DataFrame(yy).copy()
            ysim = p.concat([yhat, ysim])
            r = res['smoothed_means'].copy()
            asim = p.concat([r, asim])

        if return_states:
            return ysim, asim
        else:
            return ysim

    def system_matrices(self, para, *args, **kwargs):
        """
        Evaluate CC, TT, RR, QQ, DD, ZZ, HH at `para`.

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


        """
        CC = np.atleast_1d(self.CC(para, *args, **kwargs))
        TT = np.atleast_2d(self.TT(para, *args, **kwargs))
        RR = np.atleast_2d(self.RR(para, *args, **kwargs))
        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))

        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        self.cached_system_matrices = (CC, TT, RR, QQ, DD, ZZ, HH)

        return CC, TT, RR, QQ, DD, ZZ, HH

    def abcd_representation(self, para, *args, **kwargs):
        """
        Return ABCD representation of the system, with
        A = TT, B = RR, C = ZZ @ TT, D = ZZ @ RR.

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

    def impulse_response(self, para, h=20, use_cache=False, *args, **kwargs):
        """
        State-variable impulse responses to 1‑s.d. shocks.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.

        h : int, optional
           The maximum horizon length for the impulse responses.


        Returns
        -------
        irf : dict[str, pandas.DataFrame]
            For each shock, an (h+1) x nstates DataFrame with row 0 as impact.


        Notes
        -----
        Responses are for states; to obtain observable responses, combine with
        ZZ and DD as needed.
        """
        if use_cache:
            CC, TT, RR, QQ, DD, ZZ, HH = self.cached_system_matrices
        else:
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

    def perfect_foresight(self, para, eps_path, s0=None, include_observables=True, use_cache=False, *args, **kwargs):
        """
        Deterministic perfect-foresight simulation given a shock path.

        Parameters
        ----------
        para : array-like
            Parameter vector for system matrices.
        eps_path : array-like (T x neps)
            Deterministic path for structural shocks (no measurement error).
        s0 : array-like (ns,), optional
            Initial state. Defaults to zeros.
        include_observables : bool
            If True, also return observables path via y_t = DD + ZZ s_t.

        Returns
        -------
        out : dict
            keys: 'states' (DataFrame), and optionally 'observables' (DataFrame).
        """
        if use_cache:
            CC, TT, RR, QQ, DD, ZZ, HH = self.cached_system_matrices
        else:
            CC, TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        eps_path = np.atleast_2d(np.asarray(eps_path))
        T = eps_path.shape[0]
        ns = TT.shape[0]
        nobs = ZZ.shape[0]

        if s0 is None:
            s = np.zeros((ns,), dtype=float)
        else:
            s = np.asarray(s0, dtype=float).reshape((ns,))

        states = np.zeros((T, ns), dtype=float)
        ys = np.zeros((T, nobs), dtype=float) if include_observables else None

        for t in range(T):
            e_t = eps_path[t]
            s = CC + TT.dot(s) + RR.dot(e_t)
            states[t, :] = s
            if include_observables:
                ys[t, :] = (DD.T + ZZ.dot(s)).reshape((nobs,))

        out = {
            'states': p.DataFrame(states, columns=self.state_names),
        }
        if include_observables:
            out['observables'] = p.DataFrame(ys, columns=self.obs_names)
        return out

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
        self.cached_lre_matrices = None

    def solve_LRE(self, para, anticipated_h=0, use_cache=False, *args, **kwargs):
        qz_criterium = float(kwargs.pop("qz_criterium", 1 + 1e-6))
        return_diagnostics = bool(kwargs.pop("return_diagnostics", False))
        scale_equations = bool(kwargs.pop("scale_equations", False))
        realsmall = kwargs.pop("realsmall", None)
        if realsmall is not None:
            realsmall = float(realsmall)

        if use_cache and self.cached_lre_matrices is not None:
            G0, G1, PSI, PPI = self.cached_lre_matrices
        else:
            G0 = self.GAM0(para, *args, **kwargs)
            G1 = self.GAM1(para, *args, **kwargs)
            PSI = self.PSI(para, *args, **kwargs)
            PPI = self.PPI(para, *args, **kwargs)
            self.cached_lre_matrices = (G0, G1, PSI, PPI)

        G0 = np.atleast_2d(G0)
        G1 = np.atleast_2d(G1)
        PSI = np.atleast_2d(PSI)
        PPI = np.atleast_2d(PPI)
        C0 = np.zeros(G0.shape[0])

        nstates, nshocks = PSI.shape
        if anticipated_h > 0:
            additional_states = nshocks*anticipated_h

            G0_ext = np.eye(nstates + additional_states)
            G1_ext = np.zeros((nstates + additional_states, nstates + additional_states))
            PSI_ext = np.zeros((nstates + additional_states, nshocks + nshocks))
            PPI_ext = np.zeros((nstates + additional_states, PPI.shape[1]))

            
            # Fill the extended G0 and G1 matrices
            G0_ext[:nstates, :nstates] = G0
            G1_ext[:nstates, :nstates] = G1
            PSI_ext[:nstates, :nshocks] = PSI
            PSI_ext[nstates:(nstates+nshocks), nshocks:] = np.eye(nshocks)
            G1_ext[:nstates, -nshocks:] = PSI

            for i in range(anticipated_h - 1):
                G1_ext[nstates+(i+1)*nshocks:nstates+(i+2)*nshocks, nstates+i*nshocks:nstates+(i+1)*nshocks] = np.eye(nshocks)
            PPI_ext[:nstates, :] = PPI

            G0, G1, PSI, PPI = G0_ext, G1_ext, PSI_ext, PPI_ext
            
        nf = PPI.shape[1]

        if nf > 0:
            if scale_equations:
                # Row scaling: multiply each equation by 1 / row_norm to improve conditioning.
                # This does not change the model solution (it is a left-multiplication of all matrices),
                # but can materially improve numerical stability for large models.
                combo = np.c_[G0, G1, PSI, PPI]
                row_norm = np.linalg.norm(combo, axis=1)
                row_norm[row_norm == 0.0] = 1.0
                s = (1.0 / row_norm).reshape(-1, 1)
                G0 = s * G0
                G1 = s * G1
                PSI = s * PSI
                PPI = s * PPI

            if return_diagnostics:
                TT, RR, RC, diag = gensys(
                    G0,
                    G1,
                    PSI,
                    PPI,
                    C0=C0,
                    DIV=qz_criterium,
                    REALSMALL=realsmall if realsmall is not None else 1e-6,
                    return_diagnostics=True,
                )
                if isinstance(diag, dict):
                    diag["qz_criterium"] = qz_criterium
                    diag["scale_equations"] = scale_equations
                    diag["realsmall"] = realsmall if realsmall is not None else 1e-6
                self.last_gensys_diagnostics = diag
            else:
                TT, RR, RC = gensys(
                    G0,
                    G1,
                    PSI,
                    PPI,
                    C0=C0,
                    DIV=qz_criterium,
                    REALSMALL=realsmall if realsmall is not None else 1e-6,
                )
            # gensys returns a 2-vector RC: [existence, uniqueness]. Treat success as both == 1.
            # Other codes (e.g. coincident zeros) should map to failure (0) for downstream callers.
            RC = int((int(RC[0]) == 1) and (int(RC[1]) == 1))
            # TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensysw.gensys.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)
        else:
            TT = np.linalg.inv(G0).dot(G1)
            RR = np.linalg.inv(G0).dot(PSI)
            RC = 1

        return TT, RR, RC

    def determinacy_report(
        self,
        para,
        qz_criteria=(1 + 1e-6,),
        anticipated_h: int = 0,
        scale_equations: bool = False,
        realsmall_criteria=(None,),
        use_cache: bool = False,
        *args,
        **kwargs,
    ):
        """
        Return a lightweight determinacy report for one or more QZ thresholds.

        This is a convenience wrapper around `solve_LRE(..., return_diagnostics=True)`
        that summarizes the key `gensys` diagnostics used for existence/uniqueness.
        """
        if isinstance(qz_criteria, (int, float, np.floating)):
            qz_list = [float(qz_criteria)]
        else:
            qz_list = [float(x) for x in qz_criteria]

        realsmall_list = list(realsmall_criteria) if isinstance(realsmall_criteria, (list, tuple)) else [realsmall_criteria]

        reports = []
        for div in qz_list:
            for rs in realsmall_list:
                TT, RR, RC = self.solve_LRE(
                    para,
                    anticipated_h=anticipated_h,
                    use_cache=use_cache,
                    qz_criterium=div,
                    realsmall=rs,
                    scale_equations=scale_equations,
                    return_diagnostics=True,
                    *args,
                    **kwargs,
                )
                diag = getattr(self, "last_gensys_diagnostics", None)

                entry = {"qz_criterium": div, "realsmall": rs, "scale_equations": bool(scale_equations), "rc": int(RC)}
                if isinstance(diag, dict):
                    eig = np.asarray(diag.get("eig", []))
                    sv_unstable = diag.get("sv_unstable", None)
                    sv_loose = diag.get("sv_loose", None)
                    entry.update(
                        {
                            "nstable": int(diag.get("nstable", 0)),
                            "nunstable": int(diag.get("nunstable", 0)),
                            "coincident_zeros": bool(diag.get("coincident_zeros", False)),
                            "n_coincident_zeros": int(diag.get("n_coincident_zeros", 0))
                            if diag.get("coincident_zeros", False)
                            else 0,
                            "alpha_scale": float(diag.get("alpha_scale")) if diag.get("alpha_scale") is not None else None,
                            "beta_scale": float(diag.get("beta_scale")) if diag.get("beta_scale") is not None else None,
                            "min_sv_unstable": float(np.min(sv_unstable)) if sv_unstable is not None and np.size(sv_unstable) else None,
                            "max_sv_loose": float(np.max(sv_loose)) if sv_loose is not None and np.size(sv_loose) else None,
                            "eig_modulus_max": float(np.max(np.abs(eig))) if eig.size else None,
                            "eig_modulus_min": float(np.min(np.abs(eig))) if eig.size else None,
                        }
                    )
                reports.append(entry)

        return {"by_qz": reports}

    def pencil_nullspace_report(
        self,
        para,
        tol: float = 1e-12,
        max_vecs: int = 3,
        scale_equations: bool = True,
        use_cache: bool = False,
        *args,
        **kwargs,
    ):
        """
        Diagnose singular pencils (`coincident zeros`) by reporting right-nullspace directions.

        Computes an SVD of `[G0; G1]` and returns the dominant contributors to the smallest
        singular vectors, labeled by `state_names`.
        """
        G0 = self.GAM0(para, *args, **kwargs) if not (use_cache and self.cached_lre_matrices is not None) else self.cached_lre_matrices[0]
        G1 = self.GAM1(para, *args, **kwargs) if not (use_cache and self.cached_lre_matrices is not None) else self.cached_lre_matrices[1]
        PSI = self.PSI(para, *args, **kwargs) if not (use_cache and self.cached_lre_matrices is not None) else self.cached_lre_matrices[2]
        PPI = self.PPI(para, *args, **kwargs) if not (use_cache and self.cached_lre_matrices is not None) else self.cached_lre_matrices[3]
        self.cached_lre_matrices = (G0, G1, PSI, PPI)

        G0 = np.atleast_2d(np.asarray(G0, dtype=float))
        G1 = np.atleast_2d(np.asarray(G1, dtype=float))
        PSI = np.atleast_2d(np.asarray(PSI, dtype=float))
        PPI = np.atleast_2d(np.asarray(PPI, dtype=float))

        if scale_equations:
            combo = np.c_[G0, G1, PSI, PPI]
            row_norm = np.linalg.norm(combo, axis=1)
            row_norm[row_norm == 0.0] = 1.0
            s = (1.0 / row_norm).reshape(-1, 1)
            G0 = s * G0
            G1 = s * G1

        A = np.vstack([G0, G1])
        # Economy SVD is enough: we only need smallest singular vectors.
        _, sv, vt = np.linalg.svd(A, full_matrices=False)
        s0 = float(sv[0]) if sv.size else 1.0
        cutoff = float(tol) * max(1.0, s0)
        null_idx = np.where(sv < cutoff)[0]

        names = list(self.state_names) if self.state_names is not None else [f"x{i}" for i in range(A.shape[1])]
        vecs = []
        V = vt.T
        for k in null_idx[: int(max_vecs)]:
            v = V[:, k]
            idx = np.argsort(np.abs(v))[::-1][:20]
            vecs.append(
                {
                    "singular_value": float(sv[k]),
                    "top": [(names[i], float(v[i])) for i in idx],
                }
            )

        return {
            "n": int(A.shape[1]),
            "rank": int(np.sum(sv >= cutoff)),
            "null_dim": int(null_idx.size),
            "cutoff": cutoff,
            "smallest_singular_values": sv[-min(10, sv.size) :].astype(float).tolist() if sv.size else [],
            "vectors": vecs,
        }

    def system_matrices(self, para, *args, **kwargs):

        TT, RR, RC = self.solve_LRE(para, *args, **kwargs)
        CC = np.zeros(TT.shape[0])

        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))
        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        if RC != 1:
            TT = np.nan * TT

        self.cached_system_matrices = (CC, TT, RR, QQ, DD, ZZ, HH)
        
        return (np.ascontiguousarray(mat) for mat in (CC, TT, RR, QQ, DD, ZZ, HH))

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

    def anticipated_impulse_response(self, para, anticipated_h=1, h=20, use_cache=False, *args, **kwargs, ):
        """
        Computes anticipated impulse response functions of model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.

        h : int, optional
           The maximum horizon length for the impulse responses.
        """
        TT, RR, RC, = self.solve_LRE(para, anticipated_h=anticipated_h, use_cache=use_cache)
        from itertools import product
        additional_shock_names = [f'{s}^{h+1}' for s in product(self.shock_names, range(anticipated_h))]

        QQ = self.QQ(para)
        neps = QQ.shape[0]
        irfs = {}
        for i in range(neps):

            At = np.zeros((TT.shape[0], h + 1))
            QQz = np.zeros_like(QQ)
            QQz[i, i] = QQ[i, i]
            cQQz = np.sqrt(QQz)

            # cQQz = np.linalg.cholesky(QQz)

            At[:, 0] = (RR[:,neps:].dot(cQQz)[:, i]).squeeze()

            for j in range(h):
                At[:, j + 1] = TT.dot(At[:, j])

            irfs[self.shock_names[i]] = p.DataFrame(At.T, columns=self.state_names+
                                                    additional_shock_names)

        return irfs



class LinearDSGEModelwithSV(LinearDSGEModel):
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
            Lambda,
            Omega,
            Omega0,
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
        self.Lambda = Lambda
        self.Omega = Omega
        self.Omega0 = Omega0

        self.t0 = t0

        self.shock_names = shock_names
        self.state_names = state_names
        self.obs_names = obs_names
        self.parameter_names = parameter_names

        self.prior = prior
