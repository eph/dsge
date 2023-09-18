import numpy as np
import sympy


from sympy.matrices import zeros
from sympy import sympify
from sympy.utilities.lambdify import lambdify

from .symbols import (Variable,
                      Equation,
                      Shock,
                      Parameter,
                      TSymbol,
                      reserved_names,
                      symbolic_context)

from .Prior import construct_prior
from .data import read_data_file
from .StateSpaceModel import LinearDSGEModel, LinearDSGEModelwithSV

from typing import List, Dict, Union

from .parse_yaml import from_dict_to_mat, construct_equation_list, find_max_lead_lag, read_yaml

class EquationList(list):
    """A container for holding a set of equations.

    Inherits the list class and includes additional argument fields for context.
    """

    def __init__(self, args: List[Equation], context: Dict[str, Union[int, float, TSymbol]] = {}):
        """
        Initialize an instance of the EquationList class.

        Args:
            args (list): A list of Equation objects.
            context (dict, optional): A dictionary containing the context for the equations. Defaults to {}.
        """
        self.context = context
        return super(EquationList, self).__init__(args)

    def __setitem__(self, key: int, value: Union[str, Equation]):
        if isinstance(value, str):
            lhs, rhs = value.split("=")

            lhs = eval(lhs, self.context)
            rhs = eval(rhs, self.context)

            value = Equation(lhs, rhs)

        return super(EquationList, self).__setitem__(key, value)


class DSGE(dict):

    max_lead = 1
    max_lag = 1

    numeric_context = {}

    def __init__(self, *kargs, **kwargs):
        super(DSGE, self).__init__(self, *kargs, **kwargs)

        fvars = []
        lvars = []

        # get forward looking variables
        for eq in self["equations"]:
            variable_too_far = [v for v in eq.atoms(Variable) if v.date > 1]
            variable_too_early = [v for v in eq.atoms(Variable) if v.date < -1]

            eq_fvars = [v for v in eq.atoms(TSymbol) if v.date > 0]
            eq_lvars = [v for v in eq.atoms(TSymbol) if v.date < 0]

            for f in eq_fvars:
                if f not in fvars: fvars.append(f)

            for l in eq_lvars:
                if l not in lvars: lvars.append(l)

        self["info"]["nstate"] = len(self.variables) + len(fvars)

        self["fvars"] = fvars
        self["fvars_lagged"] = [Parameter("__LAGGED_" + f.name) for f in fvars]
        self["lvars"] = lvars
        self["re_errors"] = [Shock("eta_" + v.name) for v in self["fvars"]]

        self["re_errors_eq"] = []
        i = 0
        for fv, lag_fv in zip(fvars, self["fvars_lagged"]):
            self["re_errors_eq"].append(
                Equation(fv(-1) - lag_fv - self["re_errors"][i], sympify(0))
            )
            i += 1

        if "make_log" in self.keys():
            self["perturb_eq"] = []
            sub_dict = dict()
            sub_dict.update(
                {v: Parameter(v.name + "ss") * sympy.exp(v) for v in self["make_log"]}
            )
            sub_dict.update(
                {
                    v(-1): Parameter(v.name + "ss") * sympy.exp(v(-1))
                    for v in self["make_log"]
                }
            )
            sub_dict.update(
                {
                    v(1): Parameter(v.name + "ss") * sympy.exp(v(1))
                    for v in self["make_log"]
                }
            )

            for eq in self.equations:
                peq = eq.subs(sub_dict)
                self["perturb_eq"].append(peq)

            self["ss_ordering"] = [Variable(v.name + "ss") for v in self["make_log"]]

        else:
            self["perturb_eq"] = self["equations"]

        # context = [(s.name, s) for s in self['par_ordering']+self['var_ordering']+self['shk_ordering']+self['para_func']]
        # context = dict(context)
        # context['log'] = sympy.log
        # context['exp'] = sympy.exp
        context = {}
        # self['pertub_eq'] = EquationList(self['perturb_eq'], context)

        return

    def __repr__(self):
        indent = "\n    "
        repr = f"""
Model name: {self.name}

Parameters: {self.parameters}

Variables: {self.variables}

Shocks: {self.shocks}

Equations:
        {indent.join([eq.__repr__() for eq in self.equations])}       
        """

        return repr

    


    @property
    def equations(self):
        return self["equations"]

    @property
    def variables(self):
        return self["var_ordering"]

    @property
    def parameters(self):
        return [str(x) for x in self["par_ordering"]]

    @property
    def shocks(self):
        return self["shk_ordering"]

    @property
    def name(self):
        return self["name"]

    @property
    def neq(self):
        return len(self["perturb_eq"])

    @property
    def neq_fort(self):
        return self.neq + self.neta

    @property
    def neta(self):
        return len(self["fvars"])

    @property
    def ns(self):
        return

    @property
    def ny(self):
        return len(self["observables"])

    @property
    def neps(self):
        return len(self["shk_ordering"])

    @property
    def npara(self):
        return len(self.parameters)

    def p0(self):
        return list(map(lambda x: self["calibration"][str(x)], self.parameters))

    def python_sims_matrices(self, matrix_format="numeric"):



        vlist = self["var_ordering"] + self["fvars"]
        llist = [var(-1) for var in self["var_ordering"]] + self["fvars_lagged"]
        slist = self["shk_ordering"]

        subs_dict = dict()
        eq_cond = self["perturb_eq"] + self["re_errors_eq"]

        sub_var = self["var_ordering"]
        subs_dict.update({v: 0 for v in sub_var})
        subs_dict.update({v(1): 0 for v in sub_var})
        subs_dict.update({v(-1): 0 for v in sub_var})

        svar = len(vlist)
        evar = len(slist)
        rvar = len(self["re_errors"])
        ovar = len(self["observables"])

        GAM0 = zeros(svar, svar)
        GAM1 = zeros(svar, svar)
        PSI = zeros(svar, evar)
        PPI = zeros(svar, rvar)


        for eq_i, eq in enumerate(eq_cond):
            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))
            #print('---------------------------------------')
            #print(f'Equation: {eq}')
            #print(f'current variables {list(curr_var)}')
            for v in curr_var:
                #print(f'\tdifferentiating wrt to {v}')
                v_j = vlist.index(v)
                GAM0[eq_i, v_j] = -(eq).set_eq_zero.diff(v).subs(subs_dict)

            past_var = filter(lambda x: x in llist, eq.atoms())
            #print(f'past variables: {list(past_var)}.')
            for v in past_var:
                deq_dv = eq.set_eq_zero.diff(v).subs(subs_dict)
                v_j = llist.index(v)
                GAM1[eq_i, v_j] = deq_dv

                #print(f'\tdifferentiating wrt to {v}')

            #print(f'GAM1: {GAM1[eq_i,:]}')

            shocks = filter(lambda x: x, eq.atoms(Shock))

            for s in shocks:
                if s not in self["re_errors"]:
                    s_j = slist.index(s)
                    PSI[eq_i, s_j] = eq.set_eq_zero.diff(s).subs(subs_dict)
                else:
                    s_j = self["re_errors"].index(s)
                    PPI[eq_i, s_j] = eq.set_eq_zero.diff(s).subs(subs_dict)


            # print "\r Differentiating equation {0} of {1}.".format(eq_i, len(eq_cond)),
        DD = zeros(ovar, 1)
        ZZ = zeros(ovar, svar)

        eq_i = 0
        for obs in self["observables"]:
            eq = self["obs_equations"][str(obs)]

            DD[eq_i, 0] = eq.subs(subs_dict)

            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))
            for v in curr_var:
                v_j = vlist.index(v)
                ZZ[eq_i, v_j] = eq.diff(v).subs(subs_dict)

            eq_i += 1

        if matrix_format == "symbolic":
            QQ = self["covariance"]
            HH = self["measurement_errors"]
            return GAM0, GAM1, PSI, PPI, QQ, DD, ZZ, HH

        subs_dict = []
        context = dict([(p, Parameter(p)) for p in self.parameters])
        context.update(reserved_names)
        context['normcdf'] = lambda x: 0.5 * (1 + sympy.erf(x / sympy.sqrt(2)))
        context_f = {}
        context_f["exp"] = np.exp

        if "helper_func" in self["__data__"]["declarations"]:
            from imp import load_source

            f = self["__data__"]["declarations"]["helper_func"]["file"]
            module = load_source("helper_func", f)
            for n in self["__data__"]["declarations"]["helper_func"]["names"]:
                context[n] = sympy.Function(n)  # getattr(module, n)
                context_f[n] = getattr(module, n)


        ss = {}

        for p in self["other_para"]:
            ss[str(p)] = eval(str(self["para_func"][p.name]), context)
            context[str(p)] = ss[str(p)]


        all_para =[self.parameters + self["other_para"]]
        GAM0 = lambdify(all_para,GAM0)
        GAM1 = lambdify(all_para,GAM1)
        PSI = lambdify(all_para,PSI)
        PPI = lambdify(all_para,PPI)
        context_f['ImmutableDenseMatrix'] =  np.array
        psi = lambdify(
            [self.parameters], [ss[str(px)] for px in self["other_para"]], modules=context_f)
       
        self.psi = psi
        def add_para_func(f):
            def wrapped_f(px):
                return f([*px, *psi(px)])
            return wrapped_f

        self.GAM0 = add_para_func(GAM0)
        self.GAM1 = add_para_func(GAM1)
        self.PSI = add_para_func(PSI)
        self.PPI = add_para_func(PPI)

        QQ = self["covariance"].subs(subs_dict)
        HH = self["measurement_errors"].subs(subs_dict)

        DD = DD.subs(subs_dict)
        ZZ = ZZ.subs(subs_dict)

        QQ = lambdify(all_para, self["covariance"])
        HH = lambdify(all_para, self["measurement_errors"])
        DD = lambdify(all_para, DD)
        ZZ = lambdify(all_para, ZZ)

        self.QQ = add_para_func(QQ)
        self.DD = add_para_func(DD)
        self.ZZ = add_para_func(ZZ)
        self.HH = add_para_func(HH)

        # if self['__data__']['declarations']['type'] == 'sv':
        #     sv = self['__data__']['equations']['sv']
        #     p0 = self.p0()
        #     QQ_obs = self.QQ(p0)
        #     nshocks = len(self.shocks)
        #     assert (QQ_obs == np.eye(nshocks)).all()
        #
        #     Lambda = from_dict_to_mat(sv['transition'], self['shk_ordering'], context)
        #     Omega = from_dict_to_mat(sv['covariance'], self['shk_ordering'], context)
        #     Omega0 = from_dict_to_mat(sv['initial_covariance'], self['shk_ordering'], context)
        #
        #     self.Lambda = add_para_func(lambdify(all_para, Lambda))
        #     self.Omega = add_para_func(lambdify(all_para, Omega))
        #     self.Omega0 = add_para_func(lambdify(all_para, Omega0))

        return GAM0, GAM1, PSI, PPI

    def compile_model(self):
        self.python_sims_matrices()

        GAM0 = self.GAM0
        GAM1 = self.GAM1
        PSI = self.PSI
        PPI = self.PPI

        QQ = self.QQ
        DD = self.DD
        ZZ = self.ZZ
        HH = self.HH

        if "observables" not in self:
            self["observables"] = self["variables"].copy()
            self["obs_equations"] = dict(self["observables"], self["observables"])

        if "data" in self["__data__"]["estimation"]:
            data = read_data_file(
                self["__data__"]["estimation"]["data"], self["observables"]
            )
        else:
            data = np.nan * np.ones((100, len(self["observables"])))

        prior = None
        if "prior" in self["__data__"]["estimation"]:
            prior = construct_prior(
                self["__data__"]["estimation"]["prior"], self.parameters
            )

        from .Prior import Prior as pri

        if 1==0:#self['__data__']['declarations']['type'] == 'sv':

            dsge = LinearDSGEModelwithSV(
                  data,
                  GAM0,
                  GAM1,
                  PSI,
                  PPI,
                  QQ,
                  DD,
                  ZZ,
                  HH,
                self.Lambda,
                self.Omega,
                self.Omega0,
                  t0=0,
                  shock_names=list(map(str, self.shocks)),
                  state_names=list(map(str, self.variables + self["fvars"])),
                  obs_names=list(map(str, self["observables"])),
                  prior=pri(prior),
                  parameter_names=self.parameters
              )
        else:
            dsge = LinearDSGEModel(
                  data,
                  GAM0,
                  GAM1,
                  PSI,
                  PPI,
                  QQ,
                  DD,
                  ZZ,
                  HH,
                  t0=0,
                  shock_names=list(map(str, self.shocks)),
                  state_names=list(map(str, self.variables + self["fvars"])),
                  obs_names=list(map(str, self["observables"])),
                  prior=pri(prior),
                  parameter_names=self.parameters
              )

            dsge.psi = self.psi

            return dsge

    def update_data_file(self, file_path, start_date=None):
        if start_date is None:
            self['__data__']['estimation']['data'] = file_path
        else:
            self['__data__']['estimation']['data'] = {'file': file_path, 'start': start_date}

        return None

    def solve_model(self, p0):

        # if self.GAM0 == None:
        self.python_sims_matrices()

        from gensys import gensys_wrapper as gensys

        G0 = self.GAM0(*p0)
        G1 = self.GAM1(*p0)
        PSI = self.PSI(*p0)
        PPI = self.PPI(*p0)
        C0 = np.zeros((G0.shape[0]))
        TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensys.call_gensys(
            G0, G1, C0, PSI, PPI, 1.00000000001
        )

        return TT, RR, RC

    def create_fortran_model(self, model_dir='__fortress_tmp', **kwargs):
        from fortress import make_smc
        from .translate import smc, write_prior_file

        smc_file = smc(self)
        model_linear = self.compile_model()

        other_files = {'data.txt': model_linear.yy,          
                       'prior.txt': 'prior.txt'}
        make_smc(smc_file, model_dir, other_files=other_files, **kwargs)                      
        write_prior_file(model_linear.prior, model_dir)           


    def fix_parameters(self, **kwargs):
        """Takes an estimated parameter from a DSGESelf
        and converts it to a calibrated one."""
        for para, value in kwargs.items():
            para = Parameter(para)
            self['par_ordering'].remove(para)
            self['other_para'].append(para)
            self['para_func'][str(para)] = value

            context_tuple = ([(p, Parameter(p)) for p in self.parameters]
                 + [(p.name, p) for p in self['other_para']])

        context = dict(context_tuple)
        context['exp'] = sympy.exp
        context['log'] = sympy.log
        context['betacdf'] = sympy.Function('betacdf')

        to_replace = [(p, eval(str(self["para_func"][p.name]), context))
            for p in self['other_para']]

        from itertools import permutations

        edges = [(i,j) for i,j in permutations(to_replace,2) 
                 if type(i[1]) not in [float,int] and i[1].has(j[0])]

        from sympy import default_sort_key, topological_sort
        edges = [(v[0],dep) for v in to_replace for dep in sympy.sympify(v[1]).atoms(Parameter) if dep in self['other_para']]

        para_func = topological_sort([self['other_para'], edges], default_sort_key)[::-1]
        self['other_para'] = para_func
        return self

    @classmethod
    def read(cls, model_file):

        model_yaml = read_yaml(model_file)

        dec, cal = model_yaml["declarations"], model_yaml["calibration"]
        name = dec["name"]

        var_ordering = [Variable(v) for v in dec["variables"]]
        par_ordering = [Parameter(v) for v in dec["parameters"]]
        shk_ordering = [Shock(v) for v in dec["shocks"]]

        if "para_func" in dec:
            other_para = [Parameter(v) for v in dec["para_func"]]
        else:
            other_para = []

        if "observables" in dec:
            observables = [Variable(v) for v in dec["observables"]]
            obs_equations = model_yaml["equations"]["observables"]
        else:
            observables = [Variable(v) for v in dec["variables"]]
            obs_equations = {v: v for v in dec['variables']}

        if "measurement_errors" in dec:
            measurement_errors = [Shock(v) for v in dec["measurement_errors"]]
        else:
            measurement_errors = None

        if "make_log" in dec:
            make_log = [Variable(v) for v in dec["make_log"]]
        else:
            make_log = []

        steady_state = [0]
        init_values = [0]

        context = {s.name: s
                   for s in var_ordering + par_ordering + shk_ordering + other_para}
        context.update(symbolic_context)

        equations = []

        if "model" in model_yaml["equations"]:
            raw_equations = model_yaml["equations"]["model"]
        else:
            raw_equations = model_yaml["equations"]
        equations = construct_equation_list(raw_equations, context)

        # ------------------------------------------------------------
        # Figure out max leads and lags
        # ------------------------------------------------------------
        (max_lead_exo,
         max_lag_exo) = find_max_lead_lag(equations, shk_ordering)

        # arbitrary lags of exogenous shocks
        for s in shk_ordering:
            if abs(max_lag_exo[s]) > 0:
                var_s = Variable(s.name + "_VAR")
                var_ordering.append(var_s)
                equations.append(Equation(var_s, s))

                subs1 = [s(-i) for i in np.arange(1, abs(max_lag_exo[s]) + 1)]
                subs2 = [var_s(-i) for i in np.arange(1, abs(max_lag_exo[s]) + 1)]
                subs_dict = dict(zip(subs1, subs2))
                equations = [eq.subs(subs_dict) for eq in equations]

        (max_lead_endo,
         max_lag_endo) = find_max_lead_lag(equations, var_ordering)


        # ------------------------------------------------------------
        # arbitrary lags/leads of endogenous variables
        # ------------------------------------------------------------
        subs_dict = dict()
        old_var = var_ordering[:]
        for v in old_var:
            # lags
            for i in np.arange(2, abs(max_lag_endo[v]) + 1):
                # for lag l need to add l-1 variable
                var_l = Variable(v.name + "_LAG" + str(i - 1))

                if i == 2:
                    var_l_1 = Variable(v.name, date=-1)
                else:
                    var_l_1 = Variable(v.name + "_LAG" + str(i - 2), date=-1)

                subs_dict[Variable(v.name, date=-i)] = var_l(-1)
                var_ordering.append(var_l)
                equations.append(Equation(var_l, var_l_1))

            # still need to do leads

        equations = [eq.subs(subs_dict) for eq in equations]

        cov = cal["covariances"]

        nshock = len(shk_ordering)
        npara = len(par_ordering)

        info = {"nshock": nshock, "npara": npara}
        QQ = from_dict_to_mat(cov, shk_ordering, context)


        #------------------------------------------------------------------
        # observation equation
        #------------------------------------------------------------------
        context["sum"] = np.sum
        context["range"] = range
        for obs in obs_equations.items():
            obs_equations[obs[0]] = eval(obs[1], context)

        me_dict = {}
        if 'measurement_errors' in model_yaml['calibration']:
            me_dict = model_yaml['calibration']['measurement_errors']

        if measurement_errors is not None:
            HH = from_dict_to_mat(me_dict, measurement_errors, context)
        else:
            HH = from_dict_to_mat(me_dict, observables, context)


        calibration = model_yaml["calibration"]["parameters"]

        if "parafunc" not in cal:
            cal["parafunc"] = {}

        model_dict = {
            "var_ordering": var_ordering,
            "par_ordering": par_ordering,
            "shk_ordering": shk_ordering,
            "other_parameters": other_para,
            "other_para": other_para,
            "para_func": cal["parafunc"],
            "calibration": calibration,
            "steady_state": steady_state,
            "init_values": init_values,
            "equations": equations,
            "covariance": QQ,
            "measurement_errors": HH,
            "meas_ordering": measurement_errors,
            "info": info,
            "make_log": make_log,
            "__data__": model_yaml,
            "name": dec["name"],
            "observables": observables,
            "obs_equations": obs_equations,
        }

        model = cls(**model_dict)
        return model
