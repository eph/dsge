#!/usr/bin/env python3
import numpy as np
import sympy



from sympy import sympify
from sympy.utilities.lambdify import lambdify

from .symbols import (Variable,
                      Shock,
                      Parameter)


from .Prior import construct_prior
from .data import read_data_file
from .StateSpaceModel import LinearDSGEModel
from .parse_yaml import from_dict_to_mat, construct_equation_list, read_yaml

from sympy.printing import fcode
from sympy.printing.fortran import FCodePrinter

from .translate import write_prior_file
from fortress import make_smc


# Define a new function
def _print_Integer(self, expr):
    if expr == 0:
        return '0.0'
    else:
        return super(FCodePrinter, self)._print_Integer(expr)

# Monkey patching FCodePrinter's _print_Integer
FCodePrinter._print_Integer = _print_Integer


fortran_template = lambda model, cmodel, k, system, p0, t0: f"""
module model_t
  use, intrinsic :: iso_fortran_env, only: wp => real64

  use gensys, only: do_gensys
  use fortress, only : fortress_lgss_model
  use fortress_prior_t, only: model_prior => prior

  implicit none

  type, public, extends(fortress_lgss_model) :: model
     integer :: nvar, nshock, nval

   contains
     procedure :: system_matrices
  end type model


  interface model
     module procedure new_model
  end interface model


contains

  type(model) function new_model() result(self)

    character(len=144) :: name, datafile, priorfile
    integer :: nobs, T, ns, npara, neps

    name = 'fhp'
    datafile = 'data.txt'
    priorfile = 'prior.txt'

    nobs = {cmodel.yy.shape[1]}
    T = {cmodel.yy.shape[0]}

    self%nvar = {len(model['variables'])}
    self%nval = {len(model['values'])}
    self%nshock = {len(model['shocks'])}
    ns = 3*self%nvar + self%nval + self%nshock
    npara = {len(model['parameters'])}
    neps = {len(model['innovations'])}

    call self%construct_model(name, datafile, priorfile, npara, nobs, T, ns, neps)

!    self%p0 = {p0}

    self%t0 = {t0}
  end function new_model

  subroutine system_matrices(self, para, error)

    class(model), intent(inout) :: self
    real(wp), intent(in) :: para(self%npara)

    integer, intent(out) :: error

    double precision :: alpha0_cycle(self%nvar, self%nvar), alpha1_cycle(self%nvar, self%nvar), beta0_cycle(self%nvar, self%nshock)
    double precision :: alphaC_cycle(self%nvar, self%nvar), alphaF_cycle(self%nvar, self%nvar), alphaB_cycle(self%nvar, self%nvar), betaS_cycle(self%nvar, self%nshock)
    double precision :: alpha0_trend(self%nvar, self%nvar), alpha1_trend(self%nvar, self%nvar), betaV_trend(self%nvar, self%nval)
    double precision :: alphaC_trend(self%nvar, self%nvar), alphaF_trend(self%nvar, self%nvar), alphaB_trend(self%nvar, self%nvar)
    double precision :: value_gammaC(self%nval, self%nval), value_gamma(self%nval, self%nval),value_Cx(self%nval, self%nvar), value_Cs(self%nval, self%nshock)
    double precision :: P(self%nshock, self%nshock), R(self%nshock, self%neps)
    double precision :: DD2(self%nobs,1)
    integer :: info

    integer :: k, nvar, nval, nshock
    double precision :: A_cycle(self%nvar, self%nvar), B_cycle(self%nvar, self%nshock)
    double precision :: A_cycle_new(self%nvar, self%nvar), B_cycle_new(self%nvar, self%nshock)
    double precision :: A_trend(self%nvar, self%nvar), B_trend(self%nvar, self%nval)
    double precision :: A_trend_new(self%nvar, self%nvar), B_trend_new(self%nvar, self%nval)
    double precision :: temp1(self%nvar, self%nvar)
    integer, dimension(self%nvar) :: ipiv
    real(8), allocatable :: work(:)
    integer :: lwork
    lwork = self%nvar * self%nvar
    allocate(work(lwork))

    error = 0

    DD2 = 0.0d0

    self%QQ = 0.0d0
    self%ZZ = 0.0d0
    self%HH = 0.0d0

    {system}

    ! Initial calculations for A_cycle, B_cycle, A_trend, B_trend using LAPACK
    call dgetrf(self%nvar, self%nvar, alpha0_cycle, self%nvar, ipiv, info)
    call dgetri(self%nvar, alpha0_cycle, self%nvar, ipiv, work, lwork, info)
    call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, alpha0_cycle, self%nvar, alpha1_cycle, self%nvar, 0.0d0, A_cycle, self%nvar)
    call dgemm('N', 'N', self%nvar, self%nshock, self%nvar, 1.0d0, alpha0_cycle, self%nvar, beta0_cycle, self%nvar, 0.0d0, B_cycle, self%nvar)

    call dgetrf(self%nvar, self%nvar, alpha0_trend, self%nvar, ipiv, info)
    call dgetri(self%nvar, alpha0_trend, self%nvar, ipiv, work, lwork, info)
    call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, alpha0_trend, self%nvar, alpha1_trend, self%nvar, 0.0d0, A_trend, self%nvar)
    call dgemm('N', 'N', self%nvar, self%nval, self%nvar, 1.0d0, alpha0_trend, self%nvar, betaV_trend, self%nvar, 0.0d0, B_trend, self%nvar)

    ! ! Main loop for k
    do k = 1, 4
    !     ! Calculations for A_cycle_new
         temp1 = 0.0d0
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, -1.0d0, alphaF_cycle, self%nvar, A_cycle, self%nvar, 1.0d0, temp1, self%nvar)
         temp1 = temp1 + alphaC_cycle
         call dgetrf(self%nvar, self%nvar, temp1, self%nvar, ipiv, info)
         call dgetri(self%nvar, temp1, self%nvar, ipiv, work, lwork, info)
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, temp1, self%nvar, alphaB_cycle, self%nvar, 0.0d0, A_cycle_new, self%nvar)
    !
    !     ! ... (continue with all other calculations)
         ! Calculations for B_cycle_new

         B_cycle_new = matmul(temp1, matmul(alphaF_cycle, matmul(B_cycle, P)) + betaS_cycle)

         ! Calculations for A_trend_new
         temp1 = 0.0d0
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, -1.0d0, alphaF_trend, self%nvar, A_trend, self%nvar, 1.0d0, temp1, self%nvar)
         temp1 = temp1 + alphaC_trend
         call dgetrf(self%nvar, self%nvar, temp1, self%nvar, ipiv, info)
         call dgetri(self%nvar, temp1, self%nvar, ipiv, work, lwork, info)
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, temp1, self%nvar, alphaB_trend, self%nvar, 0.0d0, A_trend_new, self%nvar)
         B_trend_new = matmul(temp1, matmul(alphaF_trend, B_trend))

         ! Updating variables
         A_cycle = A_cycle_new
         B_cycle = B_cycle_new
         A_trend = A_trend_new
         B_trend = B_trend_new

    !
    end do
 !   call write_array_to_file('A_cycle.txt', A_cycle)
 !   call write_array_to_file('B_cycle.txt', B_cycle)
 !   call write_array_to_file('A_trend.txt', A_trend)
 !   call write_array_to_file('B_trend.txt', B_trend)

    self%DD = DD2(:,1)

    self%TT = 0.0d0
    self%RR = 0.0d0

    ! First block row
    nvar = self%nvar
    nshock = self%nshock
    nval = self%nval

self%TT(1:nvar, 1:nvar) = matmul(matmul(B_trend, value_gamma), value_Cx)
self%TT(1:nvar, (nvar+1):2*nvar) = A_cycle
self%TT(1:nvar, (2*nvar+1):(3*nvar)) = A_trend
self%TT(1:nvar, (3*nvar+1):(3*nvar+nval)) = matmul(B_trend, value_gammaC)
self%TT(1:nvar, (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(B_cycle, P) + matmul(matmul(B_trend, value_gamma), value_Cs)

! Second block row
self%TT((nvar+1):(2*nvar), (nvar+1):(2*nvar)) = A_cycle
self%TT((nvar+1):(2*nvar), (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(B_cycle, P)

! Third block row
self%TT((2*nvar+1):(3*nvar), 1:nvar) = matmul(matmul(B_trend, value_gamma), value_Cx)
self%TT((2*nvar+1):(3*nvar), (2*nvar+1):(3*nvar)) = A_trend
self%TT((2*nvar+1):(3*nvar), (3*nvar+1):(3*nvar+nval)) = matmul(B_trend, value_gammaC)
self%TT((2*nvar+1):(3*nvar), (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(matmul(B_trend, value_gamma), value_Cs)

! Fourth block row
self%TT((3*nvar+1):(3*nvar+nval), 1:nvar) = matmul(value_gamma, value_Cx)
self%TT((3*nvar+1):(3*nvar+nval), (3*nvar+1):(3*nvar+nval)) = value_gammaC
self%TT((3*nvar+1):(3*nvar+nval), (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(value_gamma, value_Cs)

! Fifth block row
self%TT((3*nvar+nval+1):(3*nvar+nval+nshock), (3*nvar+nval+1):(3*nvar+nval+nshock)) = P

! Assuming self%RR is already initialized to zero and its dimensions are set correctly
! nvar, nshock, neps are already defined

! First block
self%RR(1:nvar, 1:self%neps) = matmul(B_cycle, R)

! Second block
self%RR(nvar+1:2*nvar, 1:self%neps) = matmul(B_cycle, R)

! Third block
! Already initialized to zero, so no operation needed for zeroS

! Fourth block
! zeroV.T @ zeroS will be zero, so no operation needed here either

! Fifth block
self%RR(3*nvar+nval+1:3*nvar+nval+nshock, 1:self%neps) = R

!call write_array_to_file('TT.txt',self%TT)
!call write_array_to_file('RR.txt',self%RR)
error=0

    self%DD = DD2(:,1)




    if (info==1) error = 0

  end subroutine system_matrices


end module model_t
"""

def make_fortran_model(model, **kwargs):
    t0 = kwargs.pop("t0", 0)
    k = kwargs.pop('k', None)

    model_file = smc(model, k=k, t0=t0)
    modelc = model.compile_model(k=k)

    r = make_smc(
        model_file,
        other_files={"data.txt": modelc.yy, "prior.txt": "prior.txt"},
        **kwargs
    )

    output_dir = kwargs.pop("output_directory", "_fortress_tmp")
    write_prior_file(modelc.prior, output_dir)
    return r


def smc(model, k=None, t0=0):
    k = model['k'] if k is None else k
    cmodel = model.compile_model(k=k)

    npara = len(model['parameters'])
    para = sympy.IndexedBase("para", shape=(npara + 1,))

    from .symbols import Parameter

    fortran_subs = dict(
        zip(
            [sympy.symbols("garbage")] + model['parameters'],
            para,
        )
    )
    fortran_subs[0] = 0.0
    fortran_subs[1] = 1.0
    fortran_subs[100] = 100.0
    fortran_subs[2] = 2.0
    fortran_subs[400] = 400.0
    fortran_subs[4] = 4.0

    context_tuple = ([(p.name, p) for p in model['parameters']]
    + [(p.name, p) for p in model["other_para"].keys()])


    context = dict(context_tuple)
    context["exp"] = sympy.exp
    context["log"] = sympy.log

    to_replace = {}
    for op, expr in model["other_para"].items():
        to_replace[op] = sympify(expr, context)

    to_replace = list(to_replace.items())

    from itertools import combinations, permutations

    edges = [
        (i, j)
        for i, j in permutations(to_replace, 2)
        if type(i[1]) not in [float, int] and i[1].has(j[0])
    ]

    from sympy import default_sort_key, topological_sort

    para_func = topological_sort([to_replace, edges], default_sort_key)

    system_matrices = model.system_matrices
    to_write = ['alpha0_cycle', 'alpha1_cycle', 'beta0_cycle',
                'alphaC_cycle', 'alphaF_cycle', 'alphaB_cycle', 'betaS_cycle',
                'alpha0_trend', 'alpha1_trend', 'betaV_trend',
                'alphaC_trend', 'alphaF_trend', 'alphaB_trend',
                'value_gammaC', 'value_gamma', 'value_Cx', 'value_Cs',
                'P', 'R', 'self%QQ', 'DD2','self%ZZ','self%HH']
    print(len(system_matrices), len(to_write))
    fmats = [
        fcode(
            (mat.subs(para_func)).subs(fortran_subs),
            assign_to=n,
            source_format="free",
            standard=95,
            contract=False,
        )
        for mat, n in zip(system_matrices, to_write)
    ]
    sims_mat = "\n\n".join(fmats)
    template = fortran_template(model, cmodel, k, sims_mat, '', t0)


    return template



class LinearDSGEforFHPRepAgent(LinearDSGEModel):
    def __init__(
            self,
            yy,
            alpha0_cycle,
            alpha1_cycle,
            beta0_cycle,
            alphaC_cycle, alphaF_cycle, alphaB_cycle, betaS_cycle,
            alpha0_trend, alpha1_trend, betaV_trend,
            alphaC_trend, alphaF_trend, alphaB_trend,
            value_gammaC, value_gamma, value_Cx, value_Cs, P, R, QQ,
        DD,
        ZZ,
        HH,
        k,
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

        self.alpha0_cycle = alpha0_cycle
        self.alpha1_cycle = alpha1_cycle
        self.beta0_cycle = beta0_cycle
        self.alphaC_cycle = alphaC_cycle
        self.alphaF_cycle = alphaF_cycle
        self.alphaB_cycle = alphaB_cycle
        self.betaS_cycle = betaS_cycle
        self.alpha0_trend = alpha0_trend
        self.alpha1_trend = alpha1_trend
        self.betaV_trend = betaV_trend
        self.alphaC_trend = alphaC_trend
        self.alphaF_trend = alphaF_trend
        self.alphaB_trend = alphaB_trend
        self.value_gammaC = value_gammaC
        self.value_gamma = value_gamma
        self.value_Cx = value_Cx
        self.value_Cs = value_Cs

        self.P = P
        self.R = R
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
        self.k = k

    def system_matrices(self, p0):

       alpha0_cycle  = self.alpha0_cycle(p0)
       alpha1_cycle  = self.alpha1_cycle(p0)
       beta0_cycle   = self.beta0_cycle(p0)
       alphaC_cycle  = self.alphaC_cycle(p0)
       alphaF_cycle  = self.alphaF_cycle(p0)
       alphaB_cycle  = self.alphaB_cycle(p0)
       betaS_cycle   = self.betaS_cycle(p0)
       alpha0_trend  = self.alpha0_trend(p0)
       alpha1_trend  = self.alpha1_trend(p0)
       betaV_trend   = self.betaV_trend(p0)
       alphaC_trend  = self.alphaC_trend(p0)
       alphaF_trend  = self.alphaF_trend(p0)
       alphaB_trend  = self.alphaB_trend(p0)
       value_gammaC  = self.value_gammaC(p0)
       value_gamma   = self.value_gamma(p0)
       value_Cx      = self.value_Cx(p0)
       value_Cs      = self.value_Cs(p0)

       P = self.P(p0)
       R = self.R(p0)
       A_cycle = np.linalg.inv(alpha0_cycle) @ alpha1_cycle
       B_cycle = np.linalg.inv(alpha0_cycle) @ beta0_cycle
       A_trend = np.linalg.inv(alpha0_trend) @ alpha1_trend
       B_trend = np.linalg.inv(alpha0_trend) @ betaV_trend

       for k in range(1, self.k+1):
           A_cycle_new = np.linalg.inv(alphaC_cycle - alphaF_cycle @ A_cycle) @ alphaB_cycle
           B_cycle_new = np.linalg.inv(alphaC_cycle - alphaF_cycle @ A_cycle) @ (alphaF_cycle @ B_cycle @ P + betaS_cycle)

           A_trend_new = np.linalg.inv(alphaC_trend - alphaF_trend @ A_trend) @ alphaB_trend
           B_trend_new = np.linalg.inv(alphaC_trend - alphaF_trend @ A_trend) @ (alphaF_trend @ B_trend)

           A_cycle = A_cycle_new
           B_cycle = B_cycle_new

           A_trend = A_trend_new
           B_trend = B_trend_new
       self.A_cycle = A_cycle
       self.B_cycle = B_cycle
       self.A_trend = A_trend
       self.B_trend = B_trend
       nx = A_cycle.shape[0]
       zero = np.zeros((nx, nx))
       zeroV = np.zeros_like(B_trend)
       nx,ns = B_cycle.shape
       zeroS = np.zeros((nx, ns))


       TT = np.r_[
           np.c_[B_trend @ value_gamma @ value_Cx, A_cycle, A_trend, B_trend @ value_gammaC, B_cycle @ P + B_trend @ value_gamma @ value_Cs],
           np.c_[zero                            , A_cycle, zero   , zeroV                 , B_cycle @ P                                   ],
           np.c_[B_trend @ value_gamma @ value_Cx, zero   , A_trend, B_trend @ value_gammaC, B_trend @ value_gamma @ value_Cs              ],
           np.c_[value_gamma @ value_Cx          , zeroV.T, zeroV.T, value_gammaC          , value_gamma @ value_Cs                        ],
           np.c_[zeroS.T                         , zeroS.T, zeroS.T, zeroS.T@zeroV         , P                                             ]]


       RR = np.r_[B_cycle @ R,
                  B_cycle @ R,
                  zeroS,
                  zeroV.T @ zeroS,
                  R]

       CC = np.zeros((TT.shape[0]))
       QQ = self.QQ(p0)
       DD = self.DD(p0)
       ZZ = self.ZZ(p0)
       HH = self.HH(p0)


       return CC, TT, RR, QQ, DD, ZZ, HH


class FHPRepAgent(dict):

    def __init___():
        pass

    def p0(self):
        return list(map(lambda x: self["calibration"]['parameters'][str(x)], self['parameters']))


    @classmethod
    def read(cls, model_file, k=None):
        model_yaml = read_yaml(model_file)

        dec = model_yaml['declarations']
        variables = [Variable(v) for v in dec['variables']]
        values = [Variable(v) for v in dec['values']]
        value_updates = [Variable(v) for v in dec['value_updates']]
        shocks = [Variable(v) for v in dec['shocks']]
        innovations = [Shock(v) for v in dec['innovations']]
        parameters = [Parameter(v) for v in dec['parameters']]

        if "para_func" in dec:
            other_para = [Parameter(v) for v in dec["para_func"]]

            other_para = {op: sympify(model_yaml['calibration']["para_func"][op.name],
                                      {str(x): x for x in parameters+other_para})
                          for op in other_para}

        else:
            other_para = {}

        if "measurement_errors" in dec:
            measurement_errors = [Shock(v) for v in dec["measurement_errors"]]
        else:
            measurement_errors = None

        context = {s.name: s
                   for s in (variables +
                          values + value_updates +
                          shocks + innovations +
                          parameters + list(other_para.keys()))}

        if "observables" in dec:
            observables = [Variable(v)  for v in dec["observables"]]
            obs_equations = {o: sympify(model_yaml["model"]["observables"][str(o)], context)
                             for o in observables}
        else:
            observables = [Variable(v) for v in dec["variables"]]
            obs_equations = {v: v for v in observables}

        # set up the equations
        yaml_eq = model_yaml['model']
        equations = {}
        if 'static' in yaml_eq:
            equations['static'] = construct_equation_list(yaml_eq['static'], context)
        else:
            equations['static'] = []

        equations['cycle'] = {'terminal': construct_equation_list(yaml_eq['cycle']['terminal'], context),
                              'plan': construct_equation_list(yaml_eq['cycle']['plan'], context)}

        assert len(equations['cycle']['terminal']) + len(equations['static']) == len(variables)
        assert len(equations['cycle']['plan']) + len(equations['static']) == len(variables)

        equations['trend'] = {'terminal': construct_equation_list(yaml_eq['trend']['terminal'], context),
                              'plan': construct_equation_list(yaml_eq['trend']['plan'], context)}

        assert len(equations['trend']['terminal']) + len(equations['static']) == len(variables)
        assert len(equations['trend']['plan']) + len(equations['static']) == len(variables)

        equations['value'] = {}
        equations['value']['function'] = construct_equation_list(yaml_eq['value']['function'], context)
        equations['value']['update'] = construct_equation_list(yaml_eq['value']['update'], context)

        equations['shocks'] = construct_equation_list(yaml_eq['shocks'], context)

        cov = model_yaml['calibration']["covariance"]
        QQ = from_dict_to_mat(cov, innovations, context)

        me_dict = {}
        if 'measurement_errors' in model_yaml['calibration']:
            me_dict = model_yaml['calibration']['measurement_errors']

        if measurement_errors is not None:
            HH = from_dict_to_mat(me_dict, measurement_errors, context)
        else:

            HH = from_dict_to_mat(me_dict, observables, context)

        model_dict = {'variables': variables,
                      'values': values,
                      'value_updates': value_updates,
                      'shocks': shocks,
                      'innovations': innovations,
                      'parameters': parameters,
                      'other_para': other_para,
                      'context': context,
                      'equations': equations,
                      'calibration': model_yaml['calibration'],
                      'estimation': model_yaml['estimation'] if 'estimation' in model_yaml else {},
                      'observables': observables,
                      'obs_equations': obs_equations,
                      'QQ': QQ,
                      'HH': HH,
                      'k': dec['k']}


        return cls(**model_dict)

    def compile_model(self, k=None):

        k = self['k'] if k is None else k

        nv = len(self['variables'])
        ns = len(self['shocks'])
        nval = len(self['values'])
        v = self['variables']
        cycle_equation = self['equations']['cycle']['terminal'] + self['equations']['static']
        self.alpha0_cycle = sympy.Matrix(nv, nv , lambda i, j: cycle_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alpha1_cycle = sympy.Matrix(nv, nv , lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.beta0_cycle = sympy.Matrix(nv, ns , lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['shocks'][j]))

        cycle_equation = self['equations']['cycle']['plan'] + self['equations']['static']
        self.alphaC_cycle = sympy.Matrix(nv, nv, lambda i, j: cycle_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alphaF_cycle = sympy.Matrix(nv, nv, lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['variables'][j](+1)))
        self.alphaB_cycle = sympy.Matrix(nv, nv, lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.betaS_cycle = sympy.Matrix(nv, ns, lambda i ,j: -cycle_equation[i].set_eq_zero.diff(self['shocks'][j]))
        trend_equation = self['equations']['trend']['terminal'] + self['equations']['static']
        self.alpha0_trend = sympy.Matrix(nv, nv , lambda i, j: trend_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alpha1_trend = sympy.Matrix(nv, nv , lambda i, j: -trend_equation[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.betaV_trend = sympy.Matrix(nv, nval , lambda i, j: -trend_equation[i].set_eq_zero.diff(self['values'][j]))

        trend_equation = self['equations']['trend']['plan'] + self['equations']['static']
        self.alphaC_trend = sympy.Matrix(nv, nv, lambda i, j: trend_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alphaF_trend = sympy.Matrix(nv, nv, lambda i, j: -trend_equation[i].set_eq_zero.diff(self['variables'][j](+1)))
        self.alphaB_trend = sympy.Matrix(nv, nv, lambda i, j: -trend_equation[i].set_eq_zero.diff(self['variables'][j](-1)))

        value_equation = self['equations']['value']['function']
        lhs = sympy.Matrix(nval, nval, lambda i, j: value_equation[i].set_eq_zero.diff(self['values'][j]))
        self.value_gammaC = lhs.inv() * sympy.Matrix(nval, nval, lambda i, j: -value_equation[i].set_eq_zero.diff(self['values'][j](-1)))
        self.value_gamma = lhs.inv() * sympy.Matrix(nval, nval, lambda i, j: -value_equation[i].set_eq_zero.diff(self['value_updates'][j](-1)))

        value_equation = self['equations']['value']['update']
        lhs = sympy.Matrix(nval, nval, lambda i, j: value_equation[i].set_eq_zero.diff(self['value_updates'][j]))
        self.value_Cx = lhs.inv() * sympy.Matrix(nval, nv, lambda i, j: -value_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.value_Cs = lhs.inv() * sympy.Matrix(nval, ns, lambda i, j: -value_equation[i].set_eq_zero.diff(self['shocks'][j]))

        exo_equation = self['equations']['shocks']
        lhs = sympy.Matrix(ns, ns, lambda i, j: exo_equation[i].set_eq_zero.diff(self['shocks'][j]))
        self.P = lhs.inv() * sympy.Matrix(ns, ns, lambda i, j: -exo_equation[i].set_eq_zero.diff(self['shocks'][j](-1)))
        self.R = lhs.inv() * sympy.Matrix(ns, ns, lambda i, j: -exo_equation[i].set_eq_zero.diff(self['innovations'][j]))

        system_matrices = [self.alpha0_cycle, self.alpha1_cycle, self.beta0_cycle,
                                self.alphaC_cycle, self.alphaF_cycle, self.alphaB_cycle, self.betaS_cycle,
                                self.alpha0_trend, self.alpha1_trend, self.betaV_trend,
                                self.alphaC_trend, self.alphaF_trend, self.alphaB_trend,
                                self.value_gammaC, self.value_gamma, self.value_Cx, self.value_Cs,
                                self.P, self.R]


        all_para = self['parameters'] + list(self['other_para'].keys())

        lambdify_system_matrices = [lambdify(all_para, s)
                                    for s in system_matrices]

        intermediate_parameters = lambdify(self['parameters'], list(self['other_para'].values()))
        def expand_intermediate_parameters(f):
            def function_with_expanded_parameters(para):
                return f(*para, *intermediate_parameters(*para))
            return function_with_expanded_parameters


        alpha0_cycle = expand_intermediate_parameters(lambdify_system_matrices[0])
        alpha1_cycle = expand_intermediate_parameters(lambdify_system_matrices[1])
        beta0_cycle = expand_intermediate_parameters(lambdify_system_matrices[2])
        alphaC_cycle = expand_intermediate_parameters(lambdify_system_matrices[3])
        alphaF_cycle = expand_intermediate_parameters(lambdify_system_matrices[4])
        alphaB_cycle = expand_intermediate_parameters(lambdify_system_matrices[5])
        betaS_cycle = expand_intermediate_parameters(lambdify_system_matrices[6])
        alpha0_trend = expand_intermediate_parameters(lambdify_system_matrices[7])
        alpha1_trend = expand_intermediate_parameters(lambdify_system_matrices[8])
        betaV_trend = expand_intermediate_parameters(lambdify_system_matrices[9])
        alphaC_trend = expand_intermediate_parameters(lambdify_system_matrices[10])
        alphaF_trend = expand_intermediate_parameters(lambdify_system_matrices[11])
        alphaB_trend = expand_intermediate_parameters(lambdify_system_matrices[12])
        value_gammaC = expand_intermediate_parameters(lambdify_system_matrices[13])
        value_gamma = expand_intermediate_parameters(lambdify_system_matrices[14])
        value_Cx = expand_intermediate_parameters(lambdify_system_matrices[15])
        value_Cs = expand_intermediate_parameters(lambdify_system_matrices[16])
        P = expand_intermediate_parameters(lambdify_system_matrices[17])
        R = expand_intermediate_parameters(lambdify_system_matrices[18])

        p0 = self.p0()

        if "data" in self["estimation"]:
            data = read_data_file(self["estimation"]["data"], self["observables"])
        else:
            data = np.nan * np.ones((100, len(self["observables"])))

        system_matrices.append(self['QQ'])
        QQ = expand_intermediate_parameters(lambdify(all_para, self['QQ']))
        nobs = len(self['observables'])
        all_obj = self['variables']+self['shocks']+self['innovations']+self['values']+self['value_updates']
        subs_dict = {}
        subs_dict.update({v: 0 for v in all_obj})
        subs_dict.update({v(-1): 0 for v in all_obj})
        subs_dict.update({v(+1): 0 for v in all_obj})
        DD = sympy.Matrix(nobs, 1, lambda i,j: self['obs_equations'][self['observables'][i]]
                          .subs(subs_dict))
        system_matrices.append(DD)
        DD = expand_intermediate_parameters(lambdify(all_para, DD))

        ZZ_variables = (self['variables']
                        + [Variable(str(v)+'_cycle') for v in self['variables']]
                        + [Variable(str(v)+'_trend') for v in self['variables']]
                        + self['values'] + self['shocks'])
        ZZ = sympy.Matrix(nobs, len(ZZ_variables), lambda i, j: self['obs_equations'][self['observables'][i]].diff(ZZ_variables[j]))
        system_matrices.append(ZZ)
        ZZ = expand_intermediate_parameters(lambdify(all_para, ZZ))
        system_matrices.append(self['HH'])
        HH = expand_intermediate_parameters(lambdify(all_para, self['HH']))



        self.system_matrices = system_matrices

        prior = None
        from .Prior import Prior as pri
        if "prior" in self["estimation"]:
            prior = construct_prior(self["estimation"]["prior"], [str(pa) for pa in self['parameters']])
            prior = pri(prior)


        shock_names = [str(x) for x in self['innovations']]
        obs_names = [str(x) for x in self['observables']]
        state_names = ([str(x) for x in self['variables']]
                     + [str(x)+'_cycle' for x in self['variables']]
                     + [str(x)+'_trend' for x in self['variables']]
                     + [str(x) for x in self['values']]
                     + [str(x) for x in self['shocks']])

        parameter_names = [str(x) for x in self['parameters']]

        linmod = LinearDSGEforFHPRepAgent(data, alpha0_cycle, alpha1_cycle,
                                          beta0_cycle, alphaC_cycle, alphaF_cycle, alphaB_cycle, betaS_cycle,
                                          alpha0_trend, alpha1_trend, betaV_trend, alphaC_trend, alphaF_trend,
                                          alphaB_trend, value_gammaC, value_gamma, value_Cx, value_Cs, P, R, QQ, DD, ZZ,
                                          HH, k, t0=0,
                                          shock_names=shock_names,
                                          state_names=state_names,
                                          obs_names=obs_names,
                                          prior=prior,
                                          parameter_names=parameter_names)
        return linmod
