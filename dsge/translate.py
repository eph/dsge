import numpy as np

import os 

template_path = os.path.join(os.path.dirname(__file__), 'templates')

fortran_files = {'smc_driver': 'smc_driver_mpi.f90', 
                 'rwmh_driver': 'rwmh_driver.f90', 
                 'blockmh_driver': 'blockmcmc.f90', 
                 'Makefile': 'Makefile_dsge'}

pdict = {'gamma': 1, 
         'beta': 2, 
         'norm': 3, 
         'inv_gamma': 4, 
         'uniform': 5}


def translate(model, output_dir='.', language='fortran'):
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

    
    if language=='fortran':
        translate_fortran(model, output_dir)
    elif language=='matlab':
        translate_matlab(model, output_dir)
    elif language=='dynare':
        raise NotImplementedError('Dynare not yet implemented.')
    elif language=='julia':
        raise NotImplementedError('Julia not yet implemented.')
    else:
        raise ValueError('Unsupported language.')

        
def write_prior_file(prior, output_dir):

    def return_stats(dist):
        if dist.name=='uniform':
            return pdict[dist.name], dist.kwds['loc'], dist.kwds['loc']+dist.kwds['scale'], 0, 0
        elif dist.name=='inv_gamma':
            return pdict[dist.name], dist.a, dist.b, 0, 0
        else:
            return pdict[dist.name], dist.stats()[0], np.sqrt(dist.stats()[1]), 0, 0

    with open(os.path.join(output_dir, 'prior.txt'), mode='w') as prior_file:
        plist = [', '.join(map(str, return_stats(pr))) 
                 for pr in prior.priors]
        prior_file.write('\n'.join(plist))
            
def write_trans_file(prior, output_dir):

    def return_trans(dist):
        if dist.name=='uniform':
            return 1, dist.kwds['loc'], dist.kwds['loc']+dist.kwds['scale'], 1
        elif dist.name=='gamma' or dist.name=='inv_gamma':
            return 2, 0, 999, 1
        elif dist.name=='norm':
            return 0, -999, 999, 1
        elif dist.name=='beta':
            return 1, 0, 0.9999, 1
        else:
            raise ValueError("Unable to determine parameter value.")

    with open(os.path.join(output_dir, 'trans.txt'), mode='w') as trans_file:
        plist = [', '.join(map(str, return_trans(pr))) 
                 for pr in prior.priors]
        trans_file.write('\n'.join(plist))


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


def write_model_file(model, output_dir, language='fortran', nobs=None):
    
    system_matrices = model.python_sims_matrices(matrix_format='symbolic')
    GAM0, GAM1, PSI, PPI, QQ, DD, ZZ, HH = system_matrices

    from FCodePrinter import fcode_double as wf
    from FCodePrinter import fcode
    from sympy import MatrixSymbol as MS
    sims_mat = '\n\n'.join([wf(eval(mat), MS(mat, *eval(mat).shape), source_format='free', standard=95)
                            for mat in ['GAM0', 'GAM1', 'PSI', 'PPI']])

    ss_mat = '\n\n'.join([wf(eval(mat), MS(mat, *eval(mat).shape), source_format='free', standard=95)
                          for mat in ['QQ', 'DD', 'ZZ', 'HH']])

    para = '\n'.join(['{:} = para({:})'.format(pa, i+1) 
                      for i, pa in enumerate(model.parameters)])

    helper = '\n'.join([wf(model['para_func'][v.name], v.name) for v in model['other_para']])

    

    import re
    cal_list = [str(model['calibration'][str(x)]) for x in model.parameters]
    cal_list = [re.sub(r'([^a-zA-Z][0-9\.]+)', r'\1d0',str_p) for str_p in cal_list]
    pmsv = 'function pmsv result(para)\n  real(wp) :: para(npara)\n\n'
    pmsv += "para = (/" + ",".join(cal_list) + "/)\n\n end function pmsv"


    out = fortran_template.format(odir=output_dir, sims_mat=sims_mat, ss_mat=ss_mat, 
                                  para_list=','.join(map(str, model.parameters)), 
                                  helper_list=','.join(map(str, model['other_para'])), 
                                  para=para+helper, neq=model.neq_fort, neps=model.neps, 
                                  neta=model.neta, nobs=nobs, ny=len(model['observables']), 
                                  npara=len(model.parameters), t0=0, extra_includes='', 
                                  deff='', pmsv=pmsv, **model)

    print(out)
    
def translate_fortran(model, output_dir):

    for driver, driver_file in fortran_files.iteritems():
        with open(os.path.join(template_path, driver_file)) as template:
            if 'Makefile' in driver_file:
                output_path = os.path.join(output_dir, 'Makefile')
            else:
                output_path = os.path.join(output_dir, os.path.basename(driver_file))
            output_file = open(output_path, mode='w')
            output_file.write(template.read().format(model=model['name']))
            output_file.close()

    compiled_model = model.compile_model()
    output_dir = os.path.join(output_dir, 'model')
    try:
        os.mkdir(output_dir)
    except:
        pass

    write_prior_file(compiled_model.prior, output_dir)
    write_trans_file(compiled_model.prior, output_dir)
    write_model_file(model, output_dir, language='fortran', nobs=compiled_model.yy.shape[0])

    np.savetxt(os.path.join(output_dir, 'yy.txt'), compiled_model.yy)
    
