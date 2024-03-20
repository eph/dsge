from fortress import make_smc

DEFAULT_MODEL_DIR = '__fortress_tmp'

def create_fortran_smc(model_file,
                       output_directory=DEFAULT_MODEL_DIR,
                       other_files={}, **kwargs):
    smc = make_smc(model_file, output_directory, other_files, **kwargs)
    return smc

