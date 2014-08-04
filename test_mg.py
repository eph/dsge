import numpy as np
import sys; sys.path.append('/mq/home/m1eph00/code/test/python_compiler')

from compile_si import read
trabandt = read('/mq/DSGE/research/alt_pricing/estimation/yaml-models/trabandt.yaml')

trabandt.construct_sys_mat()
p0 = map(lambda x: eval(str(trabandt['calibration'][str(x)])), trabandt.parameters)


A = trabandt._A(*p0)
B = trabandt._B(*p0)
C = trabandt._C(*p0)
F = trabandt._F(*p0)
G = trabandt._G(*p0)
N = trabandt._N(*p0)

Aj = lambda j: np.array(trabandt._Aj(*np.append(p0, j)), dtype=float) 
Bj = lambda j: np.array(trabandt._Bj(*np.append(p0, j)), dtype=float, order='F').copy()
Cj = lambda j: np.array(trabandt._Cj(*np.append(p0, j)), dtype=float) 
Fj = lambda j: np.array(trabandt._Fj(*np.append(p0, j)), dtype=float) 
Gj = lambda j: np.array(trabandt._Gj(*np.append(p0, j)), dtype=float) 

Ainf = trabandt._Ainf(*p0)
Binf = trabandt._Binf(*p0)
Cinf = trabandt._Cinf(*p0)
Ginf = trabandt._Ginf(*p0)
Finf = trabandt._Finf(*p0)
max_it = 100

from fortran.meyer_gohde import mg
MA_VECTOR, ALPHA, BETA, RC = mg.solve_ma_alt(A, B, C, F, G, N, Aj, Bj, Cj, Fj, Gj, Ainf, Binf, Cinf, Ginf, Finf, max_it, neq=9, neps=3)
print MA_VECTOR

irfs = trabandt.compile_model().impulse_response(p0, h=12)
