from sympy import Matrix, sympify, zeros, eye
from .symbols import Variable, Parameter, Shock, Equation
from typing import List, Tuple
from .StateSpaceModel import LinearDSGEModel
from scipy.linalg import solve_discrete_lyapunov

import numpy as np

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _as_policy_tools(policy_tools) -> List[Variable]:
    tools = _as_list(policy_tools)
    return [Variable(pt) if isinstance(pt, str) else pt for pt in tools]


def _as_policy_shocks(policy_shocks) -> List[Shock]:
    shocks = _as_list(policy_shocks)
    return [Shock(s) if isinstance(s, str) else s for s in shocks]


def _loss_uses_variable(loss_string: str, var: Variable, context: dict) -> bool:
    try:
        loss_expr = sympify(loss_string, locals=context)
    except Exception:
        # Conservative: if we cannot parse, assume it might be used.
        return True
    return var in loss_expr.atoms(Variable)


def _find_difference_variable(model, tool: Variable) -> tuple[Variable | None, Equation | None]:
    """
    Look for a model variable `d` defined as a first difference of `tool`:

        d = tool - tool(-1)

    Returns (d, eq) if found, else (None, None).
    """
    for eq in model["equations"]:
        lhs = eq.lhs
        if not isinstance(lhs, Variable) or getattr(lhs, "date", 0) != 0:
            continue
        d = lhs
        try:
            poly = eq.set_eq_zero.expand()
            if (poly - (d - tool + tool(-1))).expand() == 0:
                return d, eq
        except Exception:
            continue
    return None, None


def _handle_lagged_policy_tools(
    model,
    *,
    equations: List[Equation],
    endogenous: List[Variable],
    policy_tools: List[Variable],
    loss_string: str,
) -> tuple[List[Equation], List[Variable], List[Variable]]:
    """
    Handle lagged policy instruments in remaining constraints.

    Strategy
    --------
    - If a lagged policy tool appears only inside an unused first-difference definition (e.g. `deli = i - i(-1)`),
      and `deli` is not referenced in the loss, we drop that variable+equation.
    - If such a first-difference variable *is* referenced in the loss, we reparameterize policy tools from `i`
      to `deli` and add an accumulation constraint `i = i(-1) + deli`, while dropping the original policy rule
      equation for `i` and the definition equation for `deli`.

    This preserves Dennis-form structure (no lagged controls) while supporting the common interest-rate smoothing setup.
    """
    context = {str(v): v for v in model.variables + model["parameters"] + list(model["auxiliary_parameters"].keys())}

    lagged_tools = set()
    for eq in equations:
        for v in eq.atoms(Variable):
            if v.date < 0 and any(v.name == t.name for t in policy_tools):
                lagged_tools.add(Variable(v.name))

    if not lagged_tools:
        return equations, endogenous, policy_tools

    equations_out = list(equations)
    endogenous_out = list(endogenous)
    policy_tools_out = list(policy_tools)

    for tool0 in list(lagged_tools):
        tool = next((t for t in policy_tools_out if t.name == tool0.name), None)
        if tool is None:
            continue

        diff_var, _diff_eq = _find_difference_variable(model, tool)
        if diff_var is None:
            raise NotImplementedError(
                f"Policy tool {tool.name!r} appears lagged in constraints, but no difference variable "
                f"of the form d = {tool.name} - {tool.name}(-1) was found. "
                "Either add a difference variable (recommended) or rewrite the model to avoid lagged instruments."
            )

        uses_diff = _loss_uses_variable(loss_string, diff_var, context)

        if not uses_diff:
            # Drop the difference definition if it's not needed elsewhere.
            appears_elsewhere = any((diff_var in eq.atoms(Variable)) and (eq.lhs != diff_var) for eq in equations_out)
            if appears_elsewhere:
                raise NotImplementedError(
                    f"Difference variable {diff_var.name!r} is used outside its definition, "
                    "but the loss does not reference it. Cannot safely eliminate lagged instruments."
                )
            equations_out = [eq for eq in equations_out if eq.lhs != diff_var]
            endogenous_out = [v for v in endogenous_out if v != diff_var]
            continue

        # Reparameterize: make diff_var the policy tool and make tool endogenous.
        equations_out = [eq for eq in equations_out if eq.lhs != diff_var]
        endogenous_out = [v for v in endogenous_out if v != diff_var]
        if tool not in endogenous_out:
            endogenous_out.append(tool)

        # Add accumulation constraint for the policy instrument level.
        equations_out.append(Equation(tool, tool(-1) + diff_var))

        policy_tools_out = [diff_var if t.name == tool.name else t for t in policy_tools_out]

    # Final check: lagged policy tools should no longer appear in constraints.
    for eq in equations_out:
        for v in eq.atoms(Variable):
            if v.date < 0 and any(v.name == t.name for t in policy_tools_out):
                raise NotImplementedError(
                    "Lagged policy instruments remain after preprocessing. "
                    "Rewrite the model to avoid lagged instruments, or express smoothing via a difference variable."
                )

    return equations_out, endogenous_out, policy_tools_out


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


def write_system_in_dennis_form(
    model,
    policy_tools,
    shock,
    *,
    loss_string: str | None = None,
    return_symbols: bool = False,
):
    """
    Write a model in Dennis (2007) form.

    Notes
    -----
    Dennis-form constraints support y_{t-1} and E_t y_{t+1}, and x_t and E_t x_{t+1}, but not x_{t-1}.
    When `loss_string` is provided, we attempt to handle common interest-rate smoothing setups where
    a lagged policy tool enters only through a first-difference auxiliary variable (e.g. `deli = i - i(-1)`).
    """

    # if policy_tool isn't a list, make it one
    policy_tools = _as_policy_tools(policy_tools)
    policy_shocks = _as_policy_shocks(shock)
    if len(policy_shocks) != 1:
        raise ValueError("Dennis-form solvers require exactly one policy shock name.")
    shock = policy_shocks[0]

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


    # we also need to make an equation list, omitting the equations that are associated with policy tools
    equations = []
    for eq in model['equations']:
        if not any([v == eq.lhs for v in policy_tools]):
            equations.append(eq)

    if loss_string is not None:
        equations, endogenous, policy_tools = _handle_lagged_policy_tools(
            model, equations=equations, endogenous=endogenous, policy_tools=policy_tools, loss_string=loss_string
        )
        ny = len(endogenous)

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
    if not return_symbols:
        return A0, A1, A2, A3, A4, A5, names
    return A0, A1, A2, A3, A4, A5, names, endogenous, policy_tools, nonpolicyshocks


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

    if policy_shocks is None:
        raise ValueError("policy_shocks is required for Dennis-form solvers (e.g. 'em').")

    all_parameters = model['parameters'] + list(model['auxiliary_parameters'].keys())
    # parse the loss function
    if type(beta) == str:
        beta = sympify(beta, locals={str(p): p for p in all_parameters})

    policy_instruments = _as_policy_tools(policy_instruments)

    # write the system in Dennis form (may augment for lagged instruments)
    A0, A1, A2, A3, A4, A5, names, endogenous, policy_instruments, _nonpolicyshocks = write_system_in_dennis_form(
        model, policy_instruments, policy_shocks, loss_string=loss_string, return_symbols=True
    )

    W, Q = parse_loss(loss_string, endogenous, policy_instruments, all_parameters)

    # solve the system
    ny = len(names['y'])
    nx = len(names['x'])
    nnu = len(names['nu'])
    
    GAM0_1 = (A0.T
	      .row_join(W)
	      .row_join(zeros(ny, nx))
	      .row_join((-beta*A1).T)
	      .row_join(zeros(ny, ny))
	      .row_join(zeros(ny, nx)))
    # NOTE: sign on A3.T matters when Q != 0 (Q=0 hides this).
    GAM0_2 = ((-A3.T)
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
		zeros(nx,ny+ny).row_join(eye(nx))))))	   

    nall = 2*(nx+ny*2)

    GAM0 = model.lambdify(GAM0)
    GAM1 = model.lambdify(GAM1)
    PSI = model.lambdify(PSI)
    PI = model.lambdify(PI)
    # get slice of model['covariance'] that omits policy_shocks
    policy_shocks = [str(s) for s in _as_policy_shocks(policy_shocks)]
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

    if policy_shocks is None:
        raise ValueError("policy_shocks is required for Dennis-form solvers (e.g. 'em').")

    all_parameters = model['parameters'] + list(model['auxiliary_parameters'].keys())
    # parse the loss function
    if type(beta) == str:
        beta = sympify(beta, locals={str(p): p for p in all_parameters})

    policy_instruments = _as_policy_tools(policy_instruments)

    # write the system in Dennis form (may augment for lagged instruments)
    A0, A1, A2, A3, A4, A5, names, endogenous, policy_instruments, _nonpolicyshocks = write_system_in_dennis_form(
        model, policy_instruments, policy_shocks, loss_string=loss_string, return_symbols=True
    )

    W, Q = parse_loss(loss_string, endogenous, policy_instruments, all_parameters)

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

    policy_shocks = [str(s) for s in _as_policy_shocks(policy_shocks)]
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
