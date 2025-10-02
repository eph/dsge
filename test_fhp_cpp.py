import numpy as np
from dsge import read_yaml

fhp = read_yaml('/home/eherbst/Dropbox/code/dsge/dsge/examples/fhp/fhp.yaml')
p0 = fhp.p0()

fhp_lin = fhp.compile_model()
CC, TT, RR, QQ, DD, ZZ, HH = fhp_lin.system_matrices(p0)

cpp_mat_path = '/home/eherbst/Dropbox/code/dsge/_tmp_cpp_export/'
# compare all but CC to mats store in text files in cpp_mat_path
import os

# Load matrices from text files
def load_matrices(path):
    matrices = {}
    for filename in ['TT', 'RR', 'QQ', 'DD', 'ZZ', 'HH']:
        matrices[filename] = np.loadtxt(os.path.join(path, filename + '.txt'))
    return matrices

# Compare matrices
def compare_matrices(original_matrices, cpp_matrices):
    for name, matrix in original_matrices.items():
        if name in cpp_matrices:
            if not np.allclose(matrix, cpp_matrices[name]):
                 print(f"Mismatch found in {name}")
                 print(matrix, cpp_matrices[name])

        else:
            print(f"{name} not found in cpp matrices.")

# Load system matrices from the model
original_matrices = {
    'TT': TT,
    'RR': RR,
    'QQ': QQ,
    'DD': DD,
    'ZZ': ZZ,
    'HH': HH
}

# Load matrices from text files
cpp_matrices = load_matrices(cpp_mat_path)

# Compare the matrices
compare_matrices(original_matrices, cpp_matrices)

print(fhp_lin.prior.logpdf(p0))
