import numpy as np

import os
from dsge.translate_cpp import translate_cpp

template_path = os.path.join(os.path.dirname(__file__), "templates")

fortran_files = {
    "smc_driver": "smc_driver_mpi.f90",
    "rwmh_driver": "rwmh_driver.f90",
    "blockmh_driver": "blockmcmc.f90",
    "Makefile": "Makefile_dsge",
}

pdict = {"gamma": 1, "beta": 2, "norm": 3, "invgamma_zellner": 4, "uniform": 5}

fortran_model = """
module model_t
  use, intrinsic :: iso_fortran_env, only: wp => real64

  use gensys, only: do_gensys
  use fortress, only : fortress_lgss_model
  use fortress_prior_t, only: fortress_abstract_prior
  use fortress_prior_distributions
  use fortress_random_t, only: fortress_random

  implicit none

  type, public, extends(fortress_lgss_model) :: model
     integer :: neta

   contains
     procedure :: system_matrices
  end type model


  interface model
     module procedure new_model
  end interface model

{custom_prior_code}

  {extra_code}

  type(model) function new_model() result(self)

    character(len=144) :: name
    integer :: nobs, T, ns, npara, neps

    name = '{model[name]}'

    nobs = {yy.shape[1]}
    T = {yy.shape[0]}

    ns = {model.neq_fort}
    npara = {model.npara}
    neps = {model.neps}

    ! Allocate custom prior with hardcoded parameters
    allocate(self%prior, source=model_custom_prior())

    ! Initialize model structure (no datafile or priorfile needed)
    call self%construct_lgss_model_noprior_nodata(name, npara, nobs, T, ns, neps)

    ! Allocate and initialize hardcoded data array
    allocate(self%yy(nobs, T))
{hardcoded_data}

!    self%p0 = {p0}
    self%neta = {model.neta}
    self%t0 = {t0}
  end function new_model

  subroutine system_matrices(self, para, error)

    class(model), intent(inout) :: self
    real(wp), intent(in) :: para(self%npara)

    integer, intent(out) :: error 
    
    ! double precision, intent(out) :: TT(self%ns, self%ns), RR(self%ns, neps), QQ(neps, neps)
    ! double precision, intent(out) :: DD(self%nobs), ZZ(self%nobs, self%ns), HH(self%nobs,self%nobs)
    double precision :: DD2(self%nobs,1)
    integer :: info

    double precision :: GAM0(self%ns, self%ns), GAM1(self%ns, self%ns), C(self%ns)
    double precision :: PSI(self%ns, self%neps), PPI(self%ns, self%neta), CC(self%ns)

    ! gensys
    double precision :: fmat, fwt, ywt, gev, loose, DIV
    integer :: eu(2)

    error = 1

    GAM0 = 0.0d0
    GAM1 = 0.0d0
    PSI = 0.0d0
    PPI = 0.0d0
    C = 0.0d0

    DD2 = 0.0d0

    self%QQ = 0.0d0
    self%ZZ = 0.0d0
    self%HH = 0.0d0

    {sims_mat}

    self%DD = DD2(:,1)
    call do_gensys(self%TT, CC, self%RR, fmat, fwt, ywt, gev, eu, loose, GAM0, GAM1, C, PSI, PPI, DIV)



    info = eu(1)*eu(2)

    if (info==1) error = 0

  end subroutine system_matrices


end module model_t
"""


def smc(model, t0=0, extra_code=""):

    import sympy
    from sympy.printing import fcode

    cmodel = model.compile_model()
    template = fortran_model  # open('fortran_model.f90').read()

    # Generate custom prior code instead of writing prior.txt
    custom_prior_code = generate_custom_prior_fortran(cmodel.prior)

    # Generate hardcoded data array
    hardcoded_data = generate_hardcoded_data_fortran(cmodel.yy)

    system_matrices = model.python_sims_matrices(matrix_format="symbolic")
    npara = len(model.parameters)
    para = sympy.IndexedBase("para", shape=(npara + 1,))

    from .symbols import Parameter

    fortran_subs = dict(
        zip(
            [sympy.symbols("garbage")] + [Parameter(px) for px in model.parameters],
            para,
        )
    )
    fortran_subs[0] = 0.0
    fortran_subs[1] = 1.0
    fortran_subs[100] = 100.0
    fortran_subs[2] = 2.0
    fortran_subs[400] = 400.0
    fortran_subs[4] = 4.0

    context_tuple = [(p, Parameter(p)) for p in model.parameters] + [
        (p.name, p) for p in model["other_para"]
    ]

    context = dict(context_tuple)
    context["exp"] = sympy.exp
    context["log"] = sympy.log
    context["betacdf"] = sympy.Function('betacdf')
    user_functions = {}
    if 'external' in model["__data__"]["declarations"]:
        for n in model["__data__"]["declarations"]["external"]["names"]:
            context[n] = sympy.Function(n)  
            user_functions[n] = n


    # to_replace = {}
    # for p in model["other_para"]:
    #     to_replace[p] = parse_expression(str(model["auxiliary_parameters"][p]), context)

    to_replace = model['auxiliary_parameters']
    to_replace = list(to_replace.items())

    from itertools import permutations

    edges = [ 
        (i, j)
        for i, j in permutations(to_replace, 2)
        if type(i[1]) not in [float, int] and i[1].has(j[0])
    ]

    from sympy import default_sort_key, topological_sort

    para_func = topological_sort([to_replace, edges], default_sort_key)

    to_write = ["GAM0", "GAM1", "PSI", "PPI", "self%QQ", "DD2", "self%ZZ", "self%HH"]
    # print(fcode(system_matrices[1][7,45].subs(para_func).subs(fortran_subs), user_functions=user_function))
    
    fmats = [
        fcode(
            (mat.subs(para_func)).subs(fortran_subs),
            assign_to=n,
            source_format="free",
            standard=2008,
            contract=False,
            user_functions=user_functions,
        )
        for mat, n in zip(system_matrices, to_write)
    ]
    sims_mat = "\n\n".join(fmats)
    template = template.format(
        model=model, yy=cmodel.yy, p0="", t0=t0, sims_mat=sims_mat,
        extra_code=extra_code, custom_prior_code=custom_prior_code,
        hardcoded_data=hardcoded_data
    )

    return template


def translate(model, output_dir=".", language="fortran"):
    """

    Inputs
    ------
    model - DSGEModel 
    output_dir - directory to write the model to 
    language - language to write

    Returns
    -------
    None 
    """
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    try:
        os.mkdir(output_dir)
        print("Created directory: ")
        print("\t", output_dir)
    except:
        print("Directory already exists.")

    if language == "fortran":
        translate_fortran(model, output_dir)
    elif language == "cpp":
        translate_cpp(model, output_dir)
    elif language == "dynare":
        raise NotImplementedError("Dynare not yet implemented.")
    elif language == "julia":
        raise NotImplementedError("Julia not yet implemented.")
    else:
        raise ValueError("Unsupported language.")


def generate_custom_prior_fortran(prior):
    """
    Generate Fortran code for a custom prior type with hardcoded parameters.
    This eliminates the need for external prior.txt files.
    """
    if prior is None or prior.npara == 0:
        return ""

    # Map scipy distribution names to Fortran function calls
    prior_code_lines = []
    rvs_code_lines = []

    for i, dist in enumerate(prior.priors):
        name = dist.dist.name
        idx = i + 1  # Fortran is 1-indexed

        if name == "uniform":
            lower = float(dist.kwds["loc"])
            upper = float(dist.kwds["loc"] + dist.kwds["scale"])
            prior_code_lines.append(
                f"    lpdf = lpdf + uniform_logpdf(para({idx}), {lower}_wp, {upper}_wp)"
            )
            rvs_code_lines.append(
                f"    parasim({idx}, i) = uniform_rvs({lower}_wp, {upper}_wp, self%rn)"
            )
        elif name == "norm":
            mean = dist.mean()
            std = dist.std()
            prior_code_lines.append(
                f"    lpdf = lpdf + normal_logpdf(para({idx}), {mean}_wp, {std}_wp)"
            )
            rvs_code_lines.append(
                f"    parasim({idx}, i) = normal_rvs({mean}_wp, {std}_wp, self%rn)"
            )
        elif name == "gamma":
            mean = dist.mean()
            std = dist.std()
            prior_code_lines.append(
                f"    lpdf = lpdf + gamma_logpdf(para({idx}), {mean}_wp, {std}_wp)"
            )
            rvs_code_lines.append(
                f"    parasim({idx}, i) = gamma_rvs({mean}_wp, {std}_wp, self%rn)"
            )
        elif name == "beta":
            mean = dist.mean()
            std = dist.std()
            prior_code_lines.append(
                f"    lpdf = lpdf + beta_logpdf(para({idx}), {mean}_wp, {std}_wp)"
            )
            rvs_code_lines.append(
                f"    parasim({idx}, i) = beta_rvs({mean}_wp, {std}_wp, self%rn)"
            )
        elif name == "invgamma_zellner":
            # invgamma_zellner uses shape and scale directly
            shape, scale = dist.args
            prior_code_lines.append(
                f"    lpdf = lpdf + invgamma_logpdf(para({idx}), {float(shape)}_wp, {float(scale)}_wp)"
            )
            rvs_code_lines.append(
                f"    parasim({idx}, i) = invgamma_rvs({float(shape)}_wp, {float(scale)}_wp, self%rn)"
            )
        else:
            raise ValueError(f"Unsupported distribution type: {name}")

    # Generate the complete custom prior type
    custom_prior_code = f"""
  ! Custom prior type with hardcoded parameters
  type, extends(fortress_abstract_prior) :: model_custom_prior
    type(fortress_random) :: rn
  contains
    procedure :: rvs => model_prior_rvs
    procedure :: logpdf => model_prior_logpdf
  end type model_custom_prior

  interface model_custom_prior
    module procedure new_model_custom_prior
  end interface model_custom_prior

contains

  type(model_custom_prior) function new_model_custom_prior(seed) result(self)
    integer, optional :: seed

    self%npara = {prior.npara}
    if (present(seed)) then
      self%rn = fortress_random(seed)
    else
      self%rn = fortress_random(1848)
    end if
  end function new_model_custom_prior

  function model_prior_logpdf(self, para) result(lpdf)
    class(model_custom_prior), intent(inout) :: self
    real(wp), intent(in) :: para(self%npara)
    real(wp) :: lpdf

    lpdf = 0.0_wp
{chr(10).join(prior_code_lines)}
  end function model_prior_logpdf

  function model_prior_rvs(self, nsim, seed, rng) result(parasim)
    class(model_custom_prior), intent(inout) :: self
    integer, intent(in) :: nsim
    integer, optional :: seed
    type(fortress_random), optional, intent(inout) :: rng
    real(wp) :: parasim(self%npara, nsim)
    integer :: i

    do i = 1, nsim
{chr(10).join(rvs_code_lines)}
    end do
  end function model_prior_rvs
"""

    return custom_prior_code


def generate_hardcoded_data_fortran(yy_data):
    """
    Generate Fortran code to initialize a hardcoded data array.
    This eliminates the need for external data.txt files.

    Args:
        yy_data: numpy array of shape (T, nobs) containing the observation data

    Returns:
        Fortran code string with data array initialization

    Note:
        Fortress expects yy to be shaped (nobs, T) not (T, nobs), so we transpose
    """
    import numpy as np

    yy = np.asarray(yy_data)
    T, nobs = yy.shape

    # Transpose to match fortress convention: (nobs, T) not (T, nobs)
    yy_transposed = yy.T

    # Flatten the array in Fortran order (column-major)
    flat_data = yy_transposed.flatten(order='F')

    # Format the data values
    # Break into chunks of 5 values per line for readability
    values_per_line = 5
    lines = []
    for i in range(0, len(flat_data), values_per_line):
        chunk = flat_data[i:i+values_per_line]
        formatted = ', '.join(f'{val}_wp' for val in chunk)
        lines.append(f'      {formatted}')

    # Join with commas and line continuations
    data_init = ', &\n'.join(lines)

    return f"""
  ! Hardcoded data array (nobs={nobs}, T={T})
  self%yy = reshape([{data_init}], &
                    [{nobs}, {T}])
"""


def write_prior_file(prior, output_dir):
    def return_stats(dist):
        name = dist.dist.name
        if name == "uniform":
            return (
                pdict[name],
                dist.kwds["loc"],
                dist.kwds["loc"] + dist.kwds["scale"],
                0,
                0,
            )
        elif name == "invgamma_zellner":
            return pdict[name], *dist.args, 0, 0
        else:
            return pdict[name], dist.stats()[0], np.sqrt(dist.stats()[1]), 0, 0

    with open(os.path.join(output_dir, "prior.txt"), mode="w") as prior_file:
        plist = [", ".join(map(str, return_stats(pr))) for pr in prior.priors]
        prior_file.write("\n".join(plist))


def make_fortran_model(model, **kwargs):

    t0 = kwargs.pop("t0", 0)
    extra_code = kwargs.pop("extra_code", "")
    filter_function = kwargs.pop("filter_function", None)
    

    from fortress import make_smc


    model_file = smc(model, t0=t0, extra_code=extra_code)
    if filter_function is not None:
        model_file = filter_function(model_file)

    modelc = model.compile_model()

    r = make_smc(
        model_file,
        other_files={"data.txt": modelc.yy, "prior.txt": "prior.txt"},
        **kwargs
    )

    output_dir = kwargs.pop("output_directory", "_fortress_tmp")
    write_prior_file(modelc.prior, output_dir)
    return r


def write_trans_file(prior, output_dir):
    def return_trans(dist):
        if dist.name == "uniform":
            return 1, dist.kwds["loc"], dist.kwds["loc"] + dist.kwds["scale"], 1
        elif dist.name == "gamma" or dist.name == "inv_gamma":
            return 2, 0, 999, 1
        elif dist.name == "norm":
            return 0, -999, 999, 1
        elif dist.name == "beta":
            return 1, 0, 0.9999, 1
        else:
            raise ValueError("Unable to determine parameter value.")

    with open(os.path.join(output_dir, "trans.txt"), mode="w") as trans_file:
        plist = [", ".join(map(str, return_trans(pr))) for pr in prior.priors]
        trans_file.write("\n".join(plist))


fortran_template = """
!------------------------------------------------------------
! Automatically generated fortran file.
!------------------------------------------------------------
module {name}

  use prior
  use filter
  use gensys


  implicit none

  character(len=*), parameter :: mname = '{name}'

  character(len=*), parameter :: priorfile = '{odir}/prior.txt'
  character(len=*), parameter :: priorfile2 = ''
  character(len=*), parameter :: datafile = '{odir}/yy.txt'
  character(len=*), parameter :: transfile = '{odir}/trans.txt'

  character(len=*), parameter :: initfile = ''
  character(len=*), parameter :: initwt = ''

  character(len=*), parameter :: varfile = ''

  integer, parameter :: neq = {neq}, neta = {neta}, neps = {neps}, nobs = {nobs}, npara = {npara}, ns = {neq}, ny = {ny}
  integer, parameter :: t0 = {t0}

  double precision,  parameter :: REALLY_NEG = -1000000000000.0d0
  ! data
  double precision :: YY(ny, nobs)

  ! prior stuff
  integer :: pshape(npara), pmask(npara)
  double precision :: pmean(npara), pstdd(npara), pfix(npara)

  integer :: pshape2(npara), pmask2(npara)
  double precision :: pmean2(npara), pstdd2(npara), pfix2(npara)

  ! prtr
  double precision :: trspec(4, npara)

contains

  {extra_includes}
  include '/mq/home/m1eph00/code/fortran/base/helper_functions.f90'

  subroutine sysmat(para, TT, RR, QQ, DD, ZZ, HH, info)

    double precision, intent(in) :: para(npara)

    double precision, intent(out) :: TT(neq, neq), RR(neq, neps), QQ(neps, neps), DD(ny), ZZ(ny, neq), HH(ny,ny)
    integer, intent(out) :: info

    double precision :: GAM0(neq, neq), GAM1(neq, neq), C(neq), PSI(neq, neps), PPI(neq, neta), CC(neq)

    double precision :: {para_list}
    double precision :: {helper_list}

    ! gensys
    double precision :: fmat, fwt, ywt, gev, loose, DIV
    integer :: eu(2)

    {para}

    {deff}

    GAM0 = 0.0d0
    GAM1 = 0.0d0
    PSI = 0.0d0
    PPI = 0.0d0
    C = 0.0d0

    {sims_mat}

    call do_gensys(TT, CC, RR, fmat, fwt, ywt, gev, eu, loose, GAM0, GAM1, C, PSI, PPI, DIV)

    info = eu(1)*eu(2)

    QQ = 0.0d0
    ZZ = 0.0d0
    DD = 0.0d0
    HH = 0.0d0

    {ss_mat}

    return
  end subroutine sysmat

  {pmsv}

end module {name}
"""


def write_model_file(model, output_dir, language="fortran", nobs=None):

    system_matrices = model.python_sims_matrices(matrix_format="symbolic")
    GAM0, GAM1, PSI, PPI, QQ, DD, ZZ, HH = system_matrices

    from FCodePrinter import fcode_double as wf
    from sympy import MatrixSymbol as MS

    mats = [("GAM0", GAM0), ("GAM1", GAM1), ("PSI", PSI), ("PPI", PPI)]
    sims_mat = "\n\n".join(
        [
            wf(m, MS(n, *m.shape), source_format="free", standard=95)
            for n, m in mats
        ]
    )

    ss_mats = [("QQ", QQ), ("DD", DD), ("ZZ", ZZ), ("HH", HH)]
    ss_mat = "\n\n".join(
        [
            wf(m, MS(n, *m.shape), source_format="free", standard=95)
            for n, m in ss_mats
        ]
    )

    para = "\n".join(
        ["{:} = para({:})".format(pa, i + 1) for i, pa in enumerate(model.parameters)]
    )

    helper = "\n".join(
        [wf(model["para_func"][v.name], v.name) for v in model["other_para"]]
    )

    import re

    cal_list = [str(model["calibration"][str(x)]) for x in model.parameters]
    cal_list = [re.sub(r"([^a-zA-Z][0-9\.]+)", r"\1d0", str_p) for str_p in cal_list]
    pmsv = "function pmsv result(para)\n  real(wp) :: para(npara)\n\n"
    pmsv += "para = (/" + ",".join(cal_list) + "/)\n\n end function pmsv"

    out = fortran_template.format(
        odir=output_dir,
        sims_mat=sims_mat,
        ss_mat=ss_mat,
        para_list=",".join(map(str, model.parameters)),
        helper_list=",".join(map(str, model["other_para"])),
        para=para + helper,
        neq=model.neq_fort,
        neps=model.neps,
        neta=model.neta,
        nobs=nobs,
        ny=len(model["observables"]),
        npara=len(model.parameters),
        t0=0,
        extra_includes="",
        deff="",
        pmsv=pmsv,
        **model
    )

    print(out)


def translate_fortran(model, output_dir):

    for driver, driver_file in fortran_files.iteritems():
        with open(os.path.join(template_path, driver_file)) as template:
            if "Makefile" in driver_file:
                output_path = os.path.join(output_dir, "Makefile")
            else:
                output_path = os.path.join(output_dir, os.path.basename(driver_file))
            output_file = open(output_path, mode="w")
            output_file.write(template.read().format(model=model["name"]))
            output_file.close()

    compiled_model = model.compile_model()
    output_dir = os.path.join(output_dir, "model")
    try:
        os.mkdir(output_dir)
    except:
        pass

    write_prior_file(compiled_model.prior, output_dir)
    write_trans_file(compiled_model.prior, output_dir)
    write_model_file(
        model, output_dir, language="fortran", nobs=compiled_model.yy.shape[0]
    )

    np.savetxt(os.path.join(output_dir, "yy.txt"), compiled_model.yy)
