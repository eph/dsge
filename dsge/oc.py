from sympy import Matrix, sympify, zeros, eye
from .symbols import Variable, Parameter, Shock 
from typing import List, Tuple
from .StateSpaceModel import LinearDSGEModel
from scipy.linalg import solve_discrete_lyapunov

import numpy as np

def parse_loss(loss_string: str,
               endogenous_variables: List[Variable],
               policy_instruments: List[Variable],
               parameters_vector: List[Parameter]) -> Tuple[Matrix, Matrix]:
    """
    Parse a loss function and calculate the matrices W and Q.

    Args:
    loss_string (str): The string representation of the loss function.
    endogenous_variables (List[Variable]): List of endogenous variables.
    policy_instruments (List[Variable]): List of policy instruments.
    parameters_vector (List[Parameter]): List of parameters.

    Returns:
    Tuple[Matrix, Matrix]: Matrices W and Q calculated from the loss function.
    """

    n_endog = len(endogenous_variables)
    n_policy = len(policy_instruments)

    context = {str(v): v for v in endogenous_variables+policy_instruments+parameters_vector}
    loss = sympify(loss_string, locals=context)

    # check to make sure there are no lagged or leaded variables in the loss function
    if any([v(-1) in loss.atoms() for v in endogenous_variables]):
        raise ValueError("Lagged variables in the loss function")
    if any([v(1) in loss.atoms() for v in endogenous_variables]):
        raise ValueError("Leaded variables in the loss function")
    
    W = Matrix(n_endog, n_endog, lambda i, j: loss.diff(endogenous_variables[i]).diff(endogenous_variables[j]))
    Q = Matrix(n_policy, n_policy, lambda i, j: loss.diff(policy_instruments[i]).diff(policy_instruments[j]))

    return W, Q


def write_system_in_dennis_form(model, policy_tools, shock):

    # if policy_tool isn't a list, make it one
    if not isinstance(policy_tools, list):
        policy_tools = [policy_tools]

    # now turn policy tools into Variable, if not already
    policy_tools = [Variable(pt) if isinstance(pt, str) else pt for pt in policy_tools]

    if not isinstance(shock, Shock):
        shock = Shock(shock)

    # check that policy tools are in model['variables']
    # and also construct a list of non-policy tools variables
    endogenous = []
    npolicy = 0
    for v in model.variables:
        if v in policy_tools:
            npolicy += 1
        else:
            endogenous.append(v)

    ny = len(endogenous)
    nx = len(policy_tools)

    if npolicy != len(policy_tools):
        raise ValueError("Some policy tools are not in the model")


    # we also need to make an equation list,
    # omitting the equations that are associated with policy tools
    equations = []
    for eq in model['equations']:
        if not any([v == eq.lhs for v in policy_tools]):
            equations.append(eq)

    if len(equations) != ny:
        raise ValueError("Number of equations does not match number of endogenous variables")
    
    # check that shock is in model['shocks']
    if shock not in model.shocks:
        raise ValueError("Shock not in model")

    nonpolicyshocks = [s for s in model.shocks if s != shock]

    # y_t = 'endogenous', x_t = 'policy_tools', \nu_t = 'nonpolicyshocks'
    #     \begin{align}
    # \label{eq:model-constraints}
    # A_0 y_t = A_1 y_{t-1} + A_2 E_t y_{t+1} + A_3x_{t} + A_4E_t x_{t+1} + A_5 \nu_t, \quad t \geq 0 \mbox{ and  } y_{-1} = 0.
    # \end{align}

    nnu = len(nonpolicyshocks)
    # create the A matrices as sympy matrices

    A0 = Matrix(ny, ny, lambda i,j:  equations[i].set_eq_zero.diff(endogenous[j]))
    A1 = Matrix(ny, ny, lambda i,j:  -equations[i].set_eq_zero.diff(endogenous[j](-1)))
    A2 = Matrix(ny, ny, lambda i,j:  -equations[i].set_eq_zero.diff(endogenous[j](1)))
    A3 = Matrix(ny, nx, lambda i,j:  -equations[i].set_eq_zero.diff(policy_tools[j]))
    A4 = Matrix(ny, nx, lambda i,j:  -equations[i].set_eq_zero.diff(policy_tools[j](1)))
    A5 = Matrix(ny, nnu, lambda i,j: -equations[i].set_eq_zero.diff(nonpolicyshocks[j]))
    names = {'nu': [str(s) for s in nonpolicyshocks],
             'x': [str(v) for v in policy_tools],
             'y': [str(v) for v in endogenous]}
    return A0, A1, A2, A3, A4, A5, names


def compile_commitment(model, loss_string, policy_instruments, policy_shocks=None, beta=0.99):
    """
    Solve a linear rational expectations model with commitment using the Dennis method.

    Args:
    model (Model): A model object.
    loss_string (str): The string representation of the loss function.
    policy_instrument (Variable): The policy instrument.
    policy_shock (Shock): The policy shock.

    Returns:
    Tuple[Matrix, Matrix]: Matrices W and Q calculated from the loss function.
    """

    all_parameters = model['parameters'] + list(model['auxiliary_parameters'].keys())
    # parse the loss function
    if type(beta) == str:
        beta = sympify(beta, locals={str(p): p for p in all_parameters})

    if not isinstance(policy_instruments, list):
        policy_instruments = [policy_instruments]



    W, Q = parse_loss(loss_string, 
                      [v for v in model.variables if str(v) not in policy_instruments],
                      policy_instruments, all_parameters)

    # write the system in Dennis form
    A0, A1, A2, A3, A4, A5, names = write_system_in_dennis_form(model, policy_instruments, policy_shocks)

    # solve the system
    ny = len(names['y'])
    nx = len(names['x'])
    nnu = len(names['nu'])
    
    GAM0_1 = (A0.T
	      .row_join(W)
	      .row_join(zeros(ny, nx))
	      .row_join((beta*A1).T)
	      .row_join(zeros(ny, ny))
	      .row_join(zeros(ny, nx)))
    GAM0_2 = (A3.T
	      .row_join(zeros(nx, ny))
	      .row_join(Q)
	      .row_join(zeros(nx, ny))
	      .row_join(zeros(nx, ny))
	      .row_join(zeros(nx, nx)))
    GAM0_3 = (zeros(ny, ny)
		.row_join(A0)
		.row_join(-A3)
		.row_join(zeros(ny, ny))
		.row_join(-A2)
		.row_join(-A4))
    GAM0_4 = (eye(ny)
		.row_join(zeros(ny, ny))
		.row_join(zeros(ny, nx))
		.row_join(zeros(ny, ny))
		.row_join(zeros(ny, ny))
		.row_join(zeros(ny, nx)))
    GAM0_5 = (zeros(ny, ny)
		.row_join(eye(ny))
		.row_join(zeros(ny, nx))
		.row_join(zeros(ny, ny))
		.row_join(zeros(ny, ny))
		.row_join(zeros(ny, nx)))
    GAM0_6 = (zeros(nx, ny)
		.row_join(zeros(nx, ny))
		.row_join(eye(nx))
		.row_join(zeros(nx, ny))
		.row_join(zeros(nx, ny))
		.row_join(zeros(nx, nx)))
    GAM0 = GAM0_1.col_join(GAM0_2).col_join(GAM0_3).col_join(GAM0_4).col_join(GAM0_5).col_join(GAM0_6)
    GAM1 = (beta**-1*A2.T.row_join(zeros(ny, ny+nx+ny+ny+nx))
	    ).col_join(
		beta**-1*A4.T.row_join(zeros(nx, ny+nx+ny+ny+nx))
	    ).col_join(
	   zeros(ny, ny)
	  .row_join(A1.T)
	  .row_join(zeros(ny, nx))
	  .row_join(zeros(ny, ny))
	  .row_join(zeros(ny, ny))
	  .row_join(zeros(ny, nx))
      ).col_join(
	    zeros(ny, ny)
	    .row_join(zeros(ny, ny))
	    .row_join(zeros(ny, nx))
	    .row_join(eye(ny))
	    .row_join(zeros(ny, ny))
	    .row_join(zeros(ny, nx))
	    ).col_join(
	    zeros(ny, ny)
	    .row_join(zeros(ny, ny))
	    .row_join(zeros(ny, nx))
	    .row_join(zeros(ny, ny))
	    .row_join(eye(ny))
	    .row_join(zeros(ny, nx))
	    ).col_join(
	    zeros(nx, ny)
	    .row_join(zeros(nx, ny))
	    .row_join(zeros(nx, nx))
	    .row_join(zeros(nx, ny))
	    .row_join(zeros(nx, ny))
	    .row_join(eye(nx))
	    )
    PSI = (zeros(ny, nnu).col_join(
	zeros(nx, nnu)).col_join(
	    A5).col_join(
		zeros(ny+ny+nx, nnu)))
   
    PI = (zeros(ny+ny+nx, ny+ny+nx).col_join(
	eye(ny).row_join(zeros(ny,ny+nx)).col_join(
	    zeros(ny,ny).row_join(eye(ny)).row_join(zeros(ny,nx)).col_join(
		zeros(nx,ny+ny).row_join(eye(1))))))	   

    GAM0 = model.lambdify(GAM0)
    GAM1 = model.lambdify(GAM1)
    PSI = model.lambdify(PSI)
    PI = model.lambdify(PI)


    nall = 2*(nx+ny*2)
    # get slice of model['covariance'] that omits policy_shocks
    nonpolicyshocks = [i for i, s in enumerate(model.shocks) if str(s) not in policy_shocks]
    QQ = model.lambdify(model['covariance'][nonpolicyshocks, :][:, nonpolicyshocks])
    DD = model.lambdify(zeros(nall, 1))
    ZZ = model.lambdify(eye(nall))
    HH = model.lambdify(zeros(nall, nall))
                        
    psi = None
    state_names = ([f'lagrange_{d+1}' for d in range(ny)]
                   +names['y']+names['x'])
    state_names = state_names + [f'E{n}' for n in state_names]
    
    linmod = LinearDSGEModel(
                  np.nan*np.ones((100, nall)),
                  GAM0,
                  GAM1,
                  PSI,
                  PI,
                  QQ,
                  DD,
                  ZZ,
                  HH,
                  t0=0,
                  shock_names=names['nu'],
                  state_names=state_names,
                  obs_names=state_names,
                  prior=None,
                  parameter_names=model.parameters
              )

    return linmod

def solve_fixed_point(A0np, A1np, A2np, A3np, A4np, A5np, Wnp, Qnp, beta, alpha=0.5):
    ny,nx = A3np.shape
    nnu = A5np.shape[1]

    # initial guess for H1
    H1 = np.zeros((ny,ny))
    H2 = np.zeros((ny,nnu))
    F1 = np.zeros((nx,ny))
    F2 = np.zeros((nx,nnu))

    converged = False
    alpha = 0.5
    iters = 0
    while not converged and iters < 1000:
        D = A0np - A2np @ H1 - A4np @ F1
        P = solve_discrete_lyapunov(np.sqrt(beta)*H1.T, Wnp + beta*F1.T@Qnp@F1)
 
        F1new = -np.linalg.inv(Qnp + A3np.T @ np.linalg.inv(D.T) @ P @ np.linalg.inv(D) @ A3np) @ A3np.T @ np.linalg.inv(D) @ P @ np.linalg.inv(D) @ A1np
        F2new = -np.linalg.inv(Qnp + A3np.T @ np.linalg.inv(D.T) @ P @ np.linalg.inv(D) @ A3np) @ A3np.T @ np.linalg.inv(D) @ P @ np.linalg.inv(D) @ A5np
        H1new = np.linalg.inv(D) @ (A1np + A3np @ F1new)
        H2new = np.linalg.inv(D) @ (A5np + A3np @ F2new)
 
        if (np.allclose(H1new, H1) and
            np.allclose(H2new, H2) and
            np.allclose(F1new, F1) and
            np.allclose(F2new, F2)):
 
            converged = True
 
        else:
            H1 = alpha*H1new + (1-alpha)*H1
            H2 = alpha*H2new + (1-alpha)*H2
            F1 = alpha*F1new + (1-alpha)*F1
            F2 = alpha*F2new + (1-alpha)*F2

        iters += 1

    if not converged:
        raise ValueError("Fixed point did not converge")

    return H1, H2, F1, F2


def compile_discretion(model, loss_string, policy_instruments, policy_shocks=None, beta=0.99):
    """
    Solve a linear rational expectations model with commitment using the Dennis method.

    Args:
    model (Model): A model object.
    loss_string (str): The string representation of the loss function.
    policy_instrument (Variable): The policy instrument.
    policy_shock (Shock): The policy shock.

    Returns:
    Tuple[Matrix, Matrix]: Matrices W and Q calculated from the loss function.
    """

    all_parameters = model['parameters'] + list(model['auxiliary_parameters'].keys())
    # parse the loss function
    if type(beta) == str:
        beta = sympify(beta, locals={str(p): p for p in all_parameters})

    if not isinstance(policy_instruments, list):
        policy_instruments = [policy_instruments]



    W, Q = parse_loss(loss_string, 
                      [v for v in model.variables if str(v) not in policy_instruments],
                      policy_instruments, all_parameters)

    # write the system in Dennis form
    A0, A1, A2, A3, A4, A5, names = write_system_in_dennis_form(model, policy_instruments, policy_shocks)

    A0 = model.lambdify(A0)
    A1 = model.lambdify(A1)
    A2 = model.lambdify(A2)
    A3 = model.lambdify(A3)
    A4 = model.lambdify(A4)
    A5 = model.lambdify(A5)
    beta = model.lambdify(beta)

    W = model.lambdify(W)
    Q = model.lambdify(Q)

    state_names = names['y']+names['x']
    shock_names = names['nu']
    obs_names = state_names

    nonpolicyshocks = [i for i, s in enumerate(model.shocks) if str(s) not in policy_shocks]
    QQ = model.lambdify(model['covariance'][nonpolicyshocks, :][:, nonpolicyshocks])
    nall = len(state_names)
    DD = model.lambdify(zeros(nall, 1))
    ZZ = model.lambdify(eye(nall))
    HH = model.lambdify(zeros(nall, nall))

    linmod = LinearDSGEModel(
                  np.nan*np.ones((100, len(state_names))),
                  None,
                  None,
                  None,
                  None,
                  QQ,
                  DD,
                  ZZ,
                  HH,
                  t0=0,
                  shock_names=names['nu'],
                  state_names=state_names,
                  obs_names=state_names,
                  prior=None,
                  parameter_names=model.parameters
              )
    
    # override the default solve_LRE method
    def solve(p0, alpha=0.5):
        A0np = A0(p0)
        A1np = A1(p0)
        A2np = A2(p0)
        A3np = A3(p0)
        A4np = A4(p0)
        A5np = A5(p0)
        betanp = beta(p0)
        Wnp = W(p0)
        Qnp = Q(p0)
        try:
            H1, H2, F1, F2 = solve_fixed_point(A0np, A1np, A2np, A3np, A4np, A5np, Wnp, Qnp, betanp, alpha=alpha)
            ny,nx = A3np.shape
            nnu = A5np.shape[1]
            TT = np.zeros((ny+nx, ny+nx))
            TT[:ny, :ny] = H1
            TT[ny:, :ny] = F1
            RR = np.r_[H2, F2]
            return TT, RR, 1
        except ValueError:
            print('Discretion failed to solve!')
            return None, None, None
    linmod.solve_LRE = solve
    return linmod
# Example previously under __main__ is covered by dsge/tests/test_oc.py
