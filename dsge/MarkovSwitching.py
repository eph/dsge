import numpy as np
import pandas as p
from sympy import Matrix
from sympy import sympify
class MarkovSwitchingModel:

    def __init__(self, model1, model2, transition_matrix):
        self.model1 = model1
        self.model2 = model2
        self.transition_matrix = transition_matrix

        assert model1.variables == model2.variables
        assert model1.shocks == model2.shocks
        
        self.model_1_matrices = {}
        nv = len(model1.variables)
        neps = len(model1.shocks)
        # construct matrices
        self.model_1_matrices['C'] = self.model1.lambdify(-Matrix(nv, nv, lambda i, j: self.model1['equations'][i].set_eq_zero.diff(self.model1.variables[j])))
        self.model_1_matrices['D'] = self.model1.lambdify(Matrix(nv, neps, lambda i, j: self.model1['equations'][i].set_eq_zero.diff(self.model1.shocks[j])))
        self.model_1_matrices['F'] = self.model1.lambdify(Matrix(nv, nv, lambda i, j: self.model1['equations'][i].set_eq_zero.diff(self.model1.variables[j](+1))))
        self.model_1_matrices['B'] = self.model1.lambdify(Matrix(nv, nv, lambda i, j: self.model1['equations'][i].set_eq_zero.diff(self.model1.variables[j](-1))))

        self.model_2_matrices = {}
        # construct matrices
        self.model_2_matrices['C'] = self.model2.lambdify(-Matrix(nv, nv, lambda i, j: self.model2['equations'][i].set_eq_zero.diff(self.model2.variables[j])))
        self.model_2_matrices['D'] = self.model2.lambdify(Matrix(nv, neps, lambda i, j: self.model2['equations'][i].set_eq_zero.diff(self.model2.shocks[j])))
        self.model_2_matrices['F'] = self.model2.lambdify(Matrix(nv, nv, lambda i, j: self.model2['equations'][i].set_eq_zero.diff(self.model2.variables[j](+1))))
        self.model_2_matrices['B'] = self.model2.lambdify(Matrix(nv, nv, lambda i, j: self.model2['equations'][i].set_eq_zero.diff(self.model2.variables[j](-1))))

        self.P_temp = self.model1.lambdify(transition_matrix if type(transition_matrix)==str
                                      else sympify(transition_matrix))
        # hack
        self.P = lambda x: np.array(self.P_temp(x))
        self.Q = self.model1.lambdify(self.model1['covariance'])
    def solve_LRE(self, para1, para2=None, max_k=1000):
        if para2 is None:
            para2 = para1
        C1 = self.model_1_matrices['C'](para1)
        D1 = self.model_1_matrices['D'](para1)
        F1 = self.model_1_matrices['F'](para1)
        B1 = self.model_1_matrices['B'](para1)

        C2 = self.model_2_matrices['C'](para2)
        D2 = self.model_2_matrices['D'](para2)
        F2 = self.model_2_matrices['F'](para2)
        B2 = self.model_2_matrices['B'](para2)


        T1 = np.linalg.solve(C1, B1)
        T2 = np.linalg.solve(C2, B2)
        R1 = np.linalg.solve(C1, D1)
        R2 = np.linalg.solve(C2, D2)

        P = self.P(para1)
        p11 = P[0, 0]
        p12 = P[0, 1]
        p21 = P[1, 0]
        p22 = P[1, 1]
        
        for k in range(max_k):
            T1 = np.linalg.solve(C1 - F1 @ (p11 * T1 + p12 * T2), B1)
            T2 = np.linalg.solve(C2 - F2 @ (p21 * T1 + p22 * T2), B2)
            R1 = np.linalg.solve(C1 - F1 @ (p11 * T1 + p12 * T2), D1)
            R2 = np.linalg.solve(C2 - F2 @ (p21 * T1 + p22 * T2), D2)

        return T1, T2, R1, R2

    def impulse_response(self, para1, para2=None, initial_regime=0, h=20):
        para2 = para1 if para2 is None else para2

        T1, T2, R1, R2 = self.solve_LRE(para1)

        T = [T1, T2]
        R = [R1, R2]
        Q = self.Q(para1)
        P = self.P(para1)

        irfs = {}
        for i, shock in enumerate(self.model1.shocks):
            shock = str(shock)
            irfs[shock] = np.zeros((h, len(self.model1.variables)))
            irfs[shock][0, :] = R[initial_regime][:, i] * np.sqrt(Q[i, i])
            regime = initial_regime
            for t in range(1, h):
                regime = 0 if np.random.rand() < P[regime, 0] else 1
                irfs[shock][t, :] = T[regime] @ irfs[shock][t-1, :]

            irfs[shock] = p.DataFrame(irfs[shock], columns=[str(v) for v in self.model1.variables])
               
        return irfs

    def simulate(self, para1, para2=None, initial_regime=0, previous_states=None, nsim=100):
        if para2 is None:
            para2 = para1

        T1, T2, R1, R2 = self.solve_LRE(para1,para2)

        T = [T1, T2]
        R = [R1, R2]
        Q = self.Q(para1)
        P = self.P(para1)

        ny, neps = len(self.model1.variables), len(self.model1.shocks)
        previous_states = np.zeros(ny) if previous_states is None else previous_states

        sim = np.zeros((nsim, ny))
        regimes = np.zeros(nsim, dtype=int)

        sim[0] = T[initial_regime] @ previous_states + R[initial_regime] @ (np.random.randn(neps) * np.sqrt(Q.diagonal()))
        regime = initial_regime
        regimes[0] = regime
        for i in range(1, nsim):
            regime = 0 if np.random.rand() < P[initial_regime, 0] else 1
            sim[i] = T[regime] @ sim[i-1] + R[regime] @ (np.random.randn(neps) * np.sqrt(Q.diagonal()))
            regimes[i] = regime

        sim = p.DataFrame(sim, columns=[str(v) for v in self.model1.variables])
        sim['regime'] = regimes
        return sim


