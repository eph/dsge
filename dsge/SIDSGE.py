from __future__ import division
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from symbols import Variable, Equation, Shock, Parameter, TSymbol
from symbols import timeshift as TS
from sympy.matrices import Matrix, zeros
import re

import StateSpaceModel

class SIDSGE(dict):

    A = None

    def __init__(self, *kargs, **kwargs):
        super(SIDSGE, self).__init__(self, *kargs, **kwargs)

        for eq in self['sequations']:
            variable_too_early = [v for v in eq.atoms() if isinstance(v, Variable) and v.date < -1]

    @property
    def name(self):
        return self['name']

    @property
    def p(self):
        return self['p']

    @property
    def equations(self):
        return self['equations']

    @property
    def variables(self):
        return self.endo_variables + self.exo_variables

    @property
    def endo_variables(self):
        return self['var_ordering']

    @property
    def exo_variables(self):
        return self['exo_ordering']
        
    @property
    def parameters(self):
        return self['par_ordering']

    @property
    def shocks(self):
        return self['shk_ordering']

    @property
    def nvars(self):
        return len(self.variables)

    @property
    def nendo_vars(self):
        return len(self.endo_variables)
    
    @property
    def nobs_vars(self):
        return len(self['observables'])

    @property
    def nexo_vars(self):
        return len(self.exo_variables)

    @property
    def neqs(self):
        return len(self.equations)
    
    @property
    def npara(self):
        return len(self.parameters)

    @property
    def index(self):
        return self['index'][0]

    def __repr__(self):
        s="A DSGE model with {0} variables/equations and {1}".format(self.nvars, self.npara)
        return s


    def write_matlab_gensys_file(self, odir):

        fvars = []
        lvars = []
        # construct the gensys matrices
        for eq in self['equations']:
            eq_fvars = [v for v in eq.atoms() if isinstance(v, TSymbol) and v.date > 0]
            eq_lvars = [v for v in eq.atoms() if isinstance(v, TSymbol) and v.date < 0]

            fvars = list(set(fvars).union(eq_fvars))
            lvars = list(set(lvars).union(set(eq_lvars)))

        vlist = self['var_ordering'] + fvars
        llist = lvars
        slist = self['exo_ordering']
        self['fvars'] = fvars
        self['lvars'] = lvars
        assign_list = ''
        for parai,p in enumerate(self.parameters):
            assign_list += "{0} = para({1});\n".format(p.name, parai+1)

        if self['para_func'] is not None:

            for v in self['other_para']:
                
                pstr = str(self['para_func'][v.name])
                #pstr = re.sub(r'(^[0-9\.]+)', r'\1_wp', pstr)
                assign_list += "{0} = {1};\n".format(v.name, pstr)

        assign_list += 'neq = {0};\n'.format(len(vlist))
        assign_list += 'neps = {0};\n'.format(len(slist))
        assign_list += 'neta = {0};\n'.format(len(llist))
        assign_list += 'nobs = {0};\n'.format(len(self['observables']))

        eqi = 1
        my_str = ''
        for peq in self['equations']:
            #print "    !------------------------------------------------------------"
            #print ("    ! Equation {0}".format(str(eqi)))
            #print "    !------------------------------------------------------------"
            for v in vlist:
                
                D = peq.set_eq_zero.diff(v)

                #derivative = D.subs(subs_dict)
                derivative = D
                if not(derivative==0):
                    my_str = my_str + '    GAM0({0}, {1}) = {2};\n'.format(str(eqi), v.fortind, -derivative)

                
            for v in llist:
                derivative = peq.set_eq_zero.diff(v)#.subs(subs_dict)

                if not(derivative==0):
                    my_str = my_str + '    GAM1({0}, {1}) = {2};\n'.format(str(eqi), v.fortind, derivative)


            for v in slist:
                derivative = peq.set_eq_zero.diff(v)#.subs(subs_dict)

                if not(derivative==0):
                    my_str = my_str + '    PSI({0}, {1}) = {2};\n'.format(str(eqi), v.fortind, derivative)
                
            eqi = eqi + 1

            my_str = my_str + '\n\n\n'

        eti = 1
        for fv in self['fvars']:
            my_str = my_str + '     GAM0({n}, {v}) = 1.0;\n     GAM1({n}, {Ev}) = 1.0;\n     PPI({n}, {i}) = 1.0;\n\n\n'.format(n=eqi, v=fv(-1).fortind, Ev=fv.fortind, i=eti)
            eti += 1
            eqi += 1


        QQstr = ''
        self['covariance'] = self['QQ']
        for i in range(0, self['covariance'].shape[0]):
            for j in range(0, self['covariance'].shape[1]):

                if not(self['covariance'][i, j]==0):
                    stri = self['exo_ordering'][i].fortind
                    strj = self['exo_ordering'][i].fortind
                    QQstr = QQstr + '    QQ({0}, {1}) = {2};\n'.format(stri, strj, self['covariance'][i, j])
                
        ZZstr = ''
        DDstr = ''

        obsi = 1
        nvars = len(self['var_ordering'])
        subsdict = dict(zip(self.variables, np.zeros(nvars)))
        for o in self['observables']:
            ee = self['obs_equations'][o.name]
            DDm = ee.subs(subsdict)
            DDstr = DDstr + 'DD({0}) = {1};\n'.format(obsi, DDm)

            vi = 1
            for v in self.variables:
                z = ee.diff(v)
                if not(z==0):
                    ZZstr = ZZstr + '   ZZ({0}, {1}) = {2};\n'.format(obsi, vi, z)

                vi = vi+1

            obsi = obsi+1

        ss_mat = QQstr + ZZstr + DDstr;
        var_string = ''
        nvars = len(self['var_ordering'])
        for i in xrange(1, nvars+1):
            var_string = var_string + "    {0} = {1};\n".format(self['var_ordering'][i-1].fortind, i)
            #print var_string

        nevars = len(self['fvars'])
        for i in xrange(nvars+1, nvars+nevars+1):
            var_string = var_string + "    {0} = {1};\n".format(self['fvars'][i-nvars-1].fortind, i)


        for i in xrange(1, len(self['exo_ordering'])+1):
            var_string = var_string + "    {0} = {1};\n".format(self['exo_ordering'][i-1].fortind, i)


        gensys_file = """
function [TT, RR, QQ, DD, ZZ, HH, RC] = {mname}_sysmat(para)
{para}
p{dec}
GAM0 = zeros(neq, neq);
GAM1 = zeros(neq, neq);
C    = zeros(neq, 1);
PSI  = zeros(neq, neps);
PPI  = zeros(neq, neta);

{gensys_mat}

[TT,TC,RR,TY,M,TZ,GEV,RC] = gensys(GAM0,GAM1,C,PSI,PPI,1+1E-8);

HH = zeros(nobs, nobs);
ZZ = zeros(nobs, neq);
DD = zeros(nobs, 1);

{obs_mat}

end
"""

        w = gensys_file.format(mname=self.name, dec=var_string, para=assign_list, gensys_mat=my_str, obs_mat=ss_mat);
        w = w.replace('**', '^')
        fout = open(odir + '/' + self.name + '_sysmat.m', 'w')
        fout.write(w)
        fout.close()

    def construct_sys_mat(self):
        subs_dict = dict()

        subs_dict.update( {v:0 for v in self.variables})
        subs_dict.update( {v(-1):0 for v in self.variables})
        subs_dict.update( {v(1):0 for v in self.variables})



        def PP(x, *args):
            return x[0]

        def EXP(j):
            return lambda x: E(j, x)
     
        def E(j, x):                                                        
            if len(x.atoms()) > 1:                                          
                                                                            
                for xa in x.atoms():                                        
                    if isinstance(xa, Variable):                            
                        x = x.subs({xa:E(j, xa)})                           
                return x                                                    
            else:                                                           
                if isinstance(x, Parameter):                                
                    return x                                                
                if isinstance(x, Variable):                                 
                    return type(x)(x.name, date=x.date, exp_date=j)         

        def d_rule(i, j, order=0, varsel='var_ordering', eqsel='sequations'):
            e = self[eqsel][i]
            v = self[varsel][j]
            ee = e.lhs - e.rhs

            j = self.index
            d = ee.diff(v(order)).subs(subs_dict) + ee.diff(EXP(-j)(v(order))).subs(j, 0)
            return d

        def d_rulej(i, j, order, varsel='var_ordering'):
            e = self['sequations'][i]
            v = self[varsel][j]
            ee = e.lhs - e.rhs
            j = self.index
     
            get_order = re.search('E\[([a-zA-Z0-9 -]+)\]', ee.__str__())

            if get_order is not None:
                e_order = eval(get_order.groups(0)[0], {j.name:j})
                gap = e_order + j
                d = ee.diff(EXP(e_order)(v(order))).subs({j:j+gap})
                # if d is not sympy.S.Zero:
                #     print e_order, gap, d, v(order)
            else:
                d = 0
            return d


        #--------------------------------------------------------------------------
        # 0 Matrices
        #--------------------------------------------------------------------------
        A = zeros((self.nendo_vars, self.nendo_vars))
        B = zeros((self.nendo_vars, self.nendo_vars))
        C = zeros((self.nendo_vars, self.nendo_vars))
        F = zeros((self.nendo_vars, self.nexo_vars))
        G = zeros((self.nendo_vars, self.nexo_vars))
        N = zeros((self.nexo_vars, self.nexo_vars))

        eq_i = 0
        for eq in self['sequations']:

            fvar = filter(lambda x: (x.date > 0)*(Variable(x.name) in self['var_ordering']), eq.atoms(Variable))
            for v in fvar:
                v_j = self['var_ordering'].index(Variable(v.name))
                A[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            cvar = filter(lambda x: (x.exp_date==0)*(x.date == 0)*((Variable(x.name) in self['var_ordering'])), eq.atoms(Variable))

            for v in cvar:
                j = self.index
                v_j = self['var_ordering'].index(Variable(v.name))
                B[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) #+ eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

                
            lvar = filter(lambda x: (x.exp_date==0)*(x.date < 0)*(Variable(x.name) in self['var_ordering']), eq.atoms(Variable))
            for v in lvar:
                v_j = self['var_ordering'].index(Variable(v.name))
                C[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            fshock = filter(lambda x: (x.date > 0)*(Variable(x.name) in self['exo_ordering']), eq.atoms(Variable))
            for v in fshock:
                v_j = self['exo_ordering'].index(Variable(v.name))
                F[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            cshock = filter(lambda x: (x.date == 0)*(Variable(x.name) in self['exo_ordering']), eq.atoms(Variable))
            for v in cshock:
                v_j = self['exo_ordering'].index(Variable(v.name)) 
                G[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            eq_i += 1
            print "\r Differentiating equation {0:3d} of {1}".format(eq_i, self.nendo_vars),
            

        #A = Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rule(i, jj, order=1))
        #B = Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rule(i, jj, order=0))
        #C = Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rule(i, jj, order=-1))
        #F = Matrix(self.nendo_vars, self.nexo_vars, lambda i,jj: d_rule(i, jj, order=1, varsel='exo_ordering'))
        #G = Matrix(self.nendo_vars, self.nexo_vars, lambda i,jj: d_rule(i, jj, order=0, varsel='exo_ordering'))
        #N = Matrix(self.nexo_vars, self.nexo_vars, lambda i, jj: d_rule(i, jj, order=-1, varsel='exo_ordering', eqsel='exo_equations'))

        para_subs_dict = []
        context = dict([(p.name, p) for p in self.parameters])
        context['exp'] = sympy.exp
        context['log'] = sympy.log
        print ""
        ss = {}
        for p in self['other_para']:                                       
            ss[str(p)] = eval(str(self['para_func'][p.name]), context)
            context[str(p)] = ss[str(p)]
            print "\r Constructing substitution dictionary [{0:20s}]".format(p.name), 

        psi = lambdify(self.parameters, [ss[str(px)] for px in self['other_para']])
        def add_para_func(f):
            def wrapped_f(px):
                return f(*np.append(px, psi(*px[:self.npara])))
            return wrapped_f
        
        N = -1*N

        self.A = A
        self.B = B
        self.C = C
        self.F = F
        self.G = G
        self.N = N
        self.Q = self['QQ']



        self._A = add_para_func(lambdify(self.parameters+self['other_para'], A)) 
        self._B = add_para_func(lambdify(self.parameters+self['other_para'], B)) 
        self._C = add_para_func(lambdify(self.parameters+self['other_para'], C)) 
        self._F = add_para_func(lambdify(self.parameters+self['other_para'], F)) 
        self._G = add_para_func(lambdify(self.parameters+self['other_para'], G)) 
        self._N = add_para_func(lambdify(self.parameters+self['other_para'], N)) 
        self._Q = add_para_func(lambdify(self.parameters+self['other_para'], self['QQ'])) 
        


        def find_constant_obs(i):
            ee = self['obs_equations'][(self['observables'][i]).name]
            subsdict = dict(zip(self.variables, np.zeros(self.nvars)))
            return ee.subs(subsdict)

        self.DD = Matrix(self.nobs_vars, 1, lambda i, jj : find_constant_obs(i))

        def d_ruleobs(i, j, order=0):
            ee = self['obs_equations'][(self['observables'][i]).name]
            v = self['var_ordering'][j]
            j = self.index
            d = ee.diff(v(order))
            return d

        self.Q1 = Matrix(self.nobs_vars, self.nendo_vars, lambda i, jj: d_ruleobs(i, jj, order=0))
        self.Q2 = -1*Matrix(self.nobs_vars, self.nendo_vars, lambda i, jj: d_ruleobs(i, jj, order=-1))

        self.Qlead = []
        self.Qleadempty = np.zeros((100, 1))
        for indi in np.arange(1, 100):
            self.Qlead.append(Matrix(self.nobs_vars, self.nendo_vars, lambda i, jj: d_ruleobs(i, jj, order=indi)))
            #print not(self.Qlead[indi-1] == sympy.zeros(self.nobs_vars, self.nendo_vars)), indi
            if not(self.Qlead[indi-1] == sympy.zeros(self.nobs_vars, self.nendo_vars)):
                self.Qleadempty[indi-1, :] = indi

        self.max_lead_observable = np.max(self.Qleadempty)

        
        print "zero matrices done"
        Aj= Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rulej(i, jj, order=1))     
        Bj= Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rulej(i, jj, order=0))     
        Cj= Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rulej(i, jj, order=-1))    
        Fj = Matrix(self.nendo_vars, self.nexo_vars, lambda i,jj: d_rulej(i, jj, order=1, varsel='exo_ordering'))
        Gj = Matrix(self.nendo_vars, self.nexo_vars, lambda i,jj: d_rulej(i, jj, order=0, varsel='exo_ordering'))

        print "jth matrices done"

        self.Aj = Aj
        self.Bj = Bj
        self.Cj = Cj
        self.Fj = Fj
        self.Gj = Gj

        self._Aj = add_para_func(lambdify(self.parameters+[self.index]+self['other_para'], self.Aj)) 
        self._Bj = add_para_func(lambdify(self.parameters+[self.index]+self['other_para'], self.Bj)) 
        self._Cj = add_para_func(lambdify(self.parameters+[self.index]+self['other_para'], self.Cj)) 
        self._Fj = add_para_func(lambdify(self.parameters+[self.index]+self['other_para'], self.Fj)) 
        self._Gj = add_para_func(lambdify(self.parameters+[self.index]+self['other_para'], self.Gj)) 

        Ainf= A.copy()                                                                                                     
        Binf= B.copy()                                                                                                     
        Cinf= C.copy()                                                                                                     
        Finf= F.copy()
        Ginf= G.copy()

        for ii in np.arange(0, self.nendo_vars):                                                                                         
            for jj in np.arange(0, self.nendo_vars):                                                                                     
                ad = sympy.Sum(Aj[ii, jj], (self.index, 1, sympy.oo)).doit()
                bd = sympy.Sum(Bj[ii, jj], (self.index, 1, sympy.oo)).doit()
                cd = sympy.Sum(Cj[ii, jj], (self.index, 1, sympy.oo)).doit()
                
                # if bd != 0:
                #     print Bj[ii, jj], bd.doit(), bd
                #     print 'fsdfds'
                context = [(s.name,s) for s in
                           self['var_ordering'] + self['par_ordering'] + self['index'] +
                           self['shk_ordering'] + self['exo_ordering'] + self['other_para']]
              
                context = dict(context)                                                                                    
                                                                                                                         
                context['TS'] = TS                                                                                         
                context['EXP'] = EXP                                                                                       
                                                                                                                         
                context['inf'] = sympy.oo                                                                                  
                for f in [sympy.log, sympy.exp,                                                                            
                          sympy.sin, sympy.cos, sympy.tan,                                                                 
                          sympy.asin, sympy.acos, sympy.atan,                                                              
                          sympy.sinh, sympy.cosh, sympy.tanh,                                                              
                          sympy.pi, sympy.sign]:                                                                           
                    context[str(f)] = f                                                                                    

                context['Sum'] = sympy.Sum                                                                                 
                context['oo'] = sympy.oo                                                                                   
                context['Abs'] = sympy.Abs                                                                                 
                context['Piecewise'] = PP                                                                                  
                                              
                ad = eval(ad.__str__(), context)
                bd = eval(bd.__str__(), context)                                                                             
                cd = eval(cd.__str__(), context)
                Ainf[ii, jj] = Ainf[ii, jj] + ad
                Binf[ii, jj] = Binf[ii, jj] + bd                                                                            
                Cinf[ii, jj] = Cinf[ii, jj] + cd

        self.Ainf = Ainf
        self.Binf = Binf
        self.Cinf = Cinf
        self.Ginf = Ginf
        self.Finf = Finf

        self._Ainf = add_para_func(lambdify(self.parameters+self['other_para'], self.Ainf)) 
        self._Binf = add_para_func(lambdify(self.parameters+self['other_para'], self.Binf)) 
        self._Cinf = add_para_func(lambdify(self.parameters+self['other_para'], self.Cinf)) 
        self._Finf = add_para_func(lambdify(self.parameters+self['other_para'], self.Finf)) 
        self._Ginf = add_para_func(lambdify(self.parameters+self['other_para'], self.Ginf)) 
                
        print "System matrices constructed."

    def compile_model(self):

        A = lambda x: self._A(x)
        B = lambda x: self._B(x)
        C = lambda x: self._C(x)
        F = lambda x: self._F(x)
        G = lambda x: self._G(x)
        N = lambda x: self._N(x)
        Q = lambda x: self._Q(x)
        
        Aj = lambda x, j: self._Aj(np.append(x, j))
        Bj = lambda x, j: self._Bj(np.append(x, j))
        Cj = lambda x, j: self._Cj(np.append(x, j))
        Fj = lambda x, j: self._Fj(np.append(x, j))
        Gj = lambda x, j: self._Gj(np.append(x, j))
        
        Ainf = lambda x: self._Ainf(x)
        Binf = lambda x: self._Binf(x)
        Cinf = lambda x: self._Cinf(x)
        Finf = lambda x: self._Finf(x)
        Ginf = lambda x: self._Ginf(x)

        yy = None
        mod = StateSpaceModel.LinLagExModel(yy, A, B, C, F, G, N, Q, 
                                            Aj, Bj, Cj, Fj, Gj,
                                            Ainf, Binf, Cinf, Finf, Ginf,
                                            t0=0,
                                            shock_names=map(lambda x: str(x), self.shocks),
                                            state_names=map(lambda x: str(x), self.endo_variables), 
                                            obs_names=None)

        return mod
    def write_matlab_model(self, odir):
      
        import os

        if not os.path.isabs(odir):
            odir = os.path.join(os.getcwd(), odir)
        
        print "I\'m writing to ", odir,  "."
        
        try: 
            os.mkdir(odir)
            print "Directory created."
        except:
            print "Directory already exists"


        self.__write_matlab_file(os.path.join(odir, self.name+"_model.m"))
        self.__write_starting_file(os.path.join(odir, self.name+"_pmsv.m"))
        self.__write_dec_file(os.path.join(odir, self.name+"_dec.m"))


        datafile = self['__data__']['estimation']['data']
        data = np.genfromtxt(datafile, delimiter=',', missing_values='NaN')

        datafile = os.path.join(odir, "yy.txt")
        np.savetxt(datafile, data)


        if 'estimation' in self['__data__']:
            prior_type = ['beta', 'gamma', 'normal', 'inv_gamma', 'uniform', 'fixed']
            prior = self['__data__']['estimation']['prior']
            pfile = open(os.path.join(odir, self.name+"_prior.txt"), 'w')
            prow = ['ptype', 'pmean', 'pstdd', 'pmask', 'pfix']
            fstr = "{ptype}, {pmean}, {pstdd}, {pmask}, {pfix}\n"
            
            for par in self.parameters:
                prior_spec = self['__data__']['estimation']['prior'][par.name]
                
                ptype = prior_type.index(prior_spec[0])+1
                pmean = prior_spec[1]
                pstdd = prior_spec[2]

                odict = dict(zip(prow, [ptype, pmean, pstdd, 0, 0]))
                pfile.write(fstr.format(**odict))

            pfile.close()
            
            tfile = open(os.path.join(odir, self.name+"_trspec.txt"), 'w')
            trspec = dict();
            trspec['beta'] = [0, 0.99999]
            trspec['gamma'] = [0, 999]
            trspec['normal'] = [-999, 999]
            trspec['inv_gamma'] = [0, 999]
            trspec['uniform'] = [0, 0.99999]

            for par in self.parameters:
                prior_spec = self['__data__']['estimation']['prior'][par.name]
                tfile.write("{0}, {1}\n".format(*trspec[prior_spec[0]]))

            tfile.close()


    
    def __write_dec_file(self, fstr):
        ofile = open(fstr, 'w')
        
        vari = 1
        for v in self.variables:
            ofile.write("var.{0} = {1};\n".format( v.name, vari))
            vari += 1
        shocki = 1
        for s in self.shocks:
            ofile.write("shock.{0} = {1};\n".format(s.name, shocki))
            shocki += 1
        parai = 1
        for p in self.parameters:
            ofile.write("para.{0} = {1};\n".format(p.name, parai))
            parai += 1
        ofile.close()

    def __write_starting_file(self, fstr):
        ofile = open(fstr, 'w')
        
        ofile.write('function p = ' + self.name + '_pmsv()\n')
        for p in self.parameters:
            ofile.write('{0} = {1};\n'.format(p.name, self['calibration'][p.name]))
        parai = 1
        for p in self.parameters:
            ofile.write('p({0}) = {1};\n'.format(parai, p.name))
            parai += 1
        ofile.close()



    def __print_matrix_list(self, mat_list, wrapper_function):
        pstr = ''
        for key in mat_list:
            for i in np.arange(mat_list[key].rows):
                for j in np.arange(mat_list[key].cols):
                    if mat_list[key][i, j] != 0:
                        fstr = wrapper_function(mat_list[key][i, j])
                        pstr += "{0}({1:2d}, {2:2d}) = {3};\n".format(key, i+1, j+1, fstr)
        return pstr

    def write_dynare_module(self, odir, name=None, nlag=20):

        if name==None:
            name = self.name

        if self.A is None:
            self.construct_sys_mat()


        import os

        if not os.path.isabs(odir):
            odir = os.path.join(os.getcwd(), odir)
            
        print "I\'m write to ", odir, "."

        try:
            os.mkdir(odir)
            print "Directory created."
        except:
            print "Directory already exists."

        v = ", ".join([s.__str__() for s in si_model.variables])
        s = ", ".join([s.__str__() for s in si_model.shocks]);

        

    def write_fortran_module(self, odir, name=None):

        if name==None:
            name = self.name

        if self.A is None:
            self.construct_sys_mat()


        import os

        if not os.path.isabs(odir):
            odir = os.path.join(os.getcwd(), odir)
            
        print "I\'m write to ", odir, "."

        try:
            os.mkdir(odir)
            print "Directory created."
        except:
            print "Directory already exists."

        #------------------------------------------------------------
        # Create links to fortran directories
        #------------------------------------------------------------
        base = os.path.join(odir, 'base')
        try:
            os.symlink('/mq/home/m1eph00/code/fortran/base', base)
        except:
            print "file exists"

        #------------------------------------------------------------
        # SMC FILE
        #------------------------------------------------------------
        smcfile = open('/mq/home/m1eph00/code/test/python_compiler/templates/smc_driver_mpi.f90', 'r')
        smc_driver = smcfile.read()
        smcfile.close()

        smcfilename = os.path.join(odir, 'smc_driver_mpi.f90')
        smcfile = open(smcfilename, 'w')
        smcfile.write(smc_driver.format(**{'model': name}))
        smcfile.close()

        #------------------------------------------------------------
        # RWMH
        #------------------------------------------------------------
        rwmhfile = open('/mq/home/m1eph00/code/test/python_compiler/templates/rwmh_driver.f90', 'r')
        rwmh_driver = rwmhfile.read()
        rwmhfile.close()

        rwmhfilename = os.path.join(odir, 'rwmh_driver.f90')
        rwmhfile = open(rwmhfilename, 'w')
        rwmhfile.write(rwmh_driver.format(**{'model': name}))
        rwmhfile.close()


        #------------------------------------------------------------
        # MAKEFILE
        #------------------------------------------------------------
        makefile = open('/mq/home/m1eph00/code/test/python_compiler/templates/Makefile', 'r')
        make = makefile.read()
        makefile.close()

        makefilename = os.path.join(odir, 'Makefile')
        makefile = open(makefilename, 'w')
        makefile.write(make.format(**{'model': name}))
        makefile.close()

        mdir = os.path.join(odir, 'model')
        try:
            os.mkdir(mdir)
        except:
            print "model directory exists"

        #------------------------------------------------------------
        # write priors and transfiles
        #------------------------------------------------------------
        if 'estimation' in self['__data__']:
            prior_type = ['beta', 'gamma', 'normal', 'inv_gamma', 'uniform', 'fixed']
            prior = self['__data__']['estimation']['prior']
            pfile = open(os.path.join(mdir, "prior.txt"), 'w')
            prow = ['ptype', 'pmean', 'pstdd', 'pmask', 'pfix']
            fstr = "{ptype}, {pmean}, {pstdd}, {pmask}, {pfix}\n"
            
            for par in self.parameters:
                prior_spec = self['__data__']['estimation']['prior'][par.name]
                
                ptype = prior_type.index(prior_spec[0])+1
                pmean = prior_spec[1]
                pstdd = prior_spec[2]

                odict = dict(zip(prow, [ptype, pmean, pstdd, 0, 0]))
                pfile.write(fstr.format(**odict))

            pfile.close()
            
            tfile = open(os.path.join(mdir, "trans.txt"), 'w')
            trspec = dict();
            trspec['beta'] = [1, 0, 0.99999, 1]
            trspec['gamma'] = [2, 0, 999, 1]
            trspec['normal'] = [0, -999, 999, 1]
            trspec['inv_gamma'] = [2, 0, 999, 1]
            trspec['uniform'] = [1, 0, 0.99999, 1]

            for par in self.parameters:
                prior_spec = self['__data__']['estimation']['prior'][par.name]
                tfile.write("{0}, {1}, {2}, {3}\n".format(*trspec[prior_spec[0]]))

            tfile.close()





        #------------------------------------------------------------
        # now, the data
        #------------------------------------------------------------
        datafile = self['__data__']['estimation']['data']
        with open(datafile, 'r') as df:
            data = df.read()
            delim_dict = {}

            if data.find(',') > 0:
                delim_dict['delimiter'] = ','
                
            data = np.genfromtxt(datafile, missing_values='NaN', **delim_dict)


        #data = np.genfromtxt(datafile, delimiter=',', missing_values='NaN')
        data = data[:, :len(self['observables'])]
        nobs, ny = data.shape

        if not ny == self.nobs_vars:
            print "ERROR SECTOR 12"

        datafile = os.path.join(mdir, "yy.txt")
        np.savetxt(datafile, data)


        #------------------------------------------------------------
        # now, fortran module
        #------------------------------------------------------------
        file_dict = dict()

        from extension.fcode import fcode
        def wrapper_fortran(expr):
            fstr = fcode(expr)
            fstr = re.sub(r'([^a-zA-Z][0-9\.]+)', r'\1_wp', fstr)
            # hack
            fstr = fstr.replace('_wp_wp', '_wp')
            return fstr

        # ofile = open(fstr, 'w')
            
        # ofile.write('!-----------------------------------------------------------\n')
        # ofile.write('!\n')
        # ofile.write('! Automatically generated f90-file\n')
        # ofile.write('!\n')
        # ofile.write('!-----------------------------------------------------------\n')
        para_list = ",".join([p.__str__() for p in self.parameters])
        helper_list = ",".join([p.__str__() for p in self['para_func']])

        assign_list = ''
        
        for parai,p in enumerate(self.parameters):
            assign_list += "{0} = para({1});\n".format(p.name, parai+1)

        if self['para_func'] is not None:
            assign_list += "! Helper parameters\n"
            for v in self['other_para']:
                
                pstr = str(self['para_func'][v.name])
                #pstr = re.sub(r'(^[0-9\.]+)', r'\1_wp', pstr)
                assign_list += "{0} = {1};\n".format(v.name, re.sub(r'(^[0-9\.]+|[^a-zA-Z][0-9\.]+)', r'\1_wp',pstr))


        #------------------------------------------------------------
        # function to assign observation equations
        #------------------------------------------------------------
        nQ1 = np.array(np.array(self.Q1), np.float)
        obs_ind = []
        for i in np.arange(0, ny):
            oi = np.nonzero(nQ1[i, :])[0][0]+1
            obs_ind.append(str(oi))

        obs_ind = ", ".join(obs_ind)

        obs_sub = """
function get_obs() result(obs_ind)

integer :: obs_ind(ny)

obs_ind = (/{obs}/)

return
end function
"""
        obs_func = obs_sub.format(obs=obs_ind)
        file_dict['get_obs'] = obs_func

        meas_sub = """
function get_meas_errors(para) result(HH)

    real(wp), intent(in) :: para(npara)
    real(wp) :: HH(ny, ny)

    HH = 0.0_wp

    {HH}
return
end function get_meas_errors
        """

        HH = sympy.zeros(len(self['observables']), len(self['observables']))

        meas_cal = self['__data__']['estimation'].get('measurement_errors', None)
        for key, val in meas_cal.iteritems():

            obsi = Variable(key)
            indi = self['observables'].index(obsi)
            HH[indi, indi] = eval(val)

            
        HHstr=self.__print_matrix_list({'HH':HH}, wrapper_fortran)
        meas_sub = meas_sub.format(HH=HHstr)

        file_dict['meas_sub'] = meas_sub
        #------------------------------------------------------------
        # data treatment
        #------------------------------------------------------------
        yy_sub = """
 function get_data(para) result(yvec)

    real(wp), intent(in) :: para(npara)
    real(wp) :: yvec(nobs*ny, 1),  DD(ny, 1)

    real(wp) :: {para_list}
     
    real(wp) :: {helper_list}

    integer :: i

    {assign_para}

    DD = 0.0_wp 

    {assign_DD}

    do i = 1,nobs
        yvec((i-1)*ny+1:i*ny, 1) = YY(:, i) - DD(:, 1)
    end do

  end function get_data
"""

     
        print_dict = {'para_list': para_list, 'helper_list': helper_list, 'assign_para': assign_list}
        print_dict['assign_DD'] = self.__print_matrix_list({'DD':self.DD}, wrapper_fortran)

        file_dict['get_data'] = yy_sub.format(**print_dict)

        #------------------------------------------------------------
        # exogenous processes
        #------------------------------------------------------------
        exo_sub = """
  subroutine {sub_name}_matrix({sub_name}, para, neps)
  
    integer,  intent(in) :: neps
    real(wp), intent(in) :: para(npara)
    real(wp),  intent(out) :: {sub_name}(neps, neps)

    real(wp) :: {para_list}
     
    real(wp) :: {helper_list}
   
    {assign_para}

    {sub_name} = 0.0_wp

    {assign_mat}

  end subroutine {sub_name}_matrix
"""

        print_dict = {'para_list': para_list, 'helper_list': helper_list, 'assign_para': assign_list}
        print_dict['assign_mat'] = self.__print_matrix_list({'N':self.N}, wrapper_fortran)
        print_dict['sub_name'] = 'N'

        file_dict['N_matrix'] = exo_sub.format(**print_dict)

        print_dict = {'para_list': para_list, 'helper_list': helper_list, 'assign_para': assign_list}
        print_dict['assign_mat'] = self.__print_matrix_list({'OMEGA':self.Q}, wrapper_fortran)
        print_dict['sub_name'] = 'OMEGA'

        file_dict['OMEGA_matrix'] = exo_sub.format(**print_dict)

        #------------------------------------------------------------
        # And now, the matrices
        #------------------------------------------------------------

        mat_sub = """
  subroutine {sub_name}(A{post}, B{post}, C{post}, F{post}, G{post}, {ind}para, neq, neps, npara)
   
    integer, intent(in) :: {ind}neq, neps, npara
    real(wp), intent(in) :: para(npara)
     
    real(wp), intent(out) :: A{post}(neq, neq), B{post}(neq, neq), C{post}(neq, neq), F{post}(neq, neps), G{post}(neq, neps)
     
    real(wp) :: {para_list}
     
    real(wp) :: {helper_list}
   
    {assign_para}
   
   
    A{post} = 0.0_wp
    B{post} = 0.0_wp
    C{post} = 0.0_wp
    F{post} = 0.0_wp
    G{post} = 0.0_wp
   
    {assign_mat}

  end subroutine {sub_name}
"""





        print_dict = {'para_list': para_list, 'helper_list': helper_list, 'assign_para': assign_list}
        
        mat_list = {'A':self.A, 'B':self.B, 'C':self.C, 'F': self.F, 'G': self.G};

        print_dict['assign_mat'] = self.__print_matrix_list(mat_list, wrapper_fortran)
        print_dict['post'] = ''
        print_dict['sub_name'] = 'zero_matrices'
        print_dict['ind'] = ''


        file_dict['zero_matrices'] = mat_sub.format(**print_dict)

        mat_list = {'Aj':self.Aj, 'Bj':self.Bj, 'Cj':self.Cj, 'Fj': self.Fj, 'Gj': self.Gj};
        print_dict['assign_mat'] = self.__print_matrix_list(mat_list, wrapper_fortran)
        print_dict['sub_name'] = 'j_matrices'
        print_dict['ind'] = 'j, '
        print_dict['post'] = 'j'

        file_dict['j_matrices'] = mat_sub.format(**print_dict)

        
        mat_list = {'Ainf':self.Ainf, 'Binf':self.Binf, 'Cinf':self.Cinf, 'Finf': self.Finf, 'Ginf': self.Ginf};
        print_dict['assign_mat'] = self.__print_matrix_list(mat_list, wrapper_fortran)
        print_dict['sub_name'] = 'inf_matrices'
        print_dict['ind'] = ''
        print_dict['post'] = 'inf'

        file_dict['inf_matrices'] = mat_sub.format(**print_dict)
        file_dict['pfile'] = mdir + '/prior.txt'
        file_dict['datafile'] = mdir + '/yy.txt'
        file_dict['transfile'] = mdir + '/trans.txt'
        file_dict['name'] = name
        file_dict['neq'] = self.neqs
        file_dict['neps'] = self.nexo_vars
        file_dict['nobs'] = nobs
        file_dict['npara'] = self.npara
        file_dict['ny'] = self.nobs_vars

        cal_list = [str(self['calibration'][str(x)]) for x in self.parameters]
        cal_list = [re.sub(r'([^a-zA-Z][0-9\.]+)', r'\1_wp',str_p) for str_p in cal_list]

        pmsv = 'function pmsv result(para)\n  real(wp) :: para(npara)\n\n'
        pmsv += "para = (/" + ",".join(cal_list) + "/)\n\n end function pmsv"

        file_dict['pmsv'] = pmsv

        fortran_template = """
!------------------------------------------------------------
! Automatically generated fortran file.
!------------------------------------------------------------ 
include 'mkl_pardiso.f90'
module {name}

  use mkl95_precision, only: wp => dp
  use mkl_pardiso

  use prior

  implicit none

  character(len=*), parameter :: mname='{name}'

  character(len=*), parameter :: priorfile = '{pfile}'
  character(len=*), parameter :: datafile = '{datafile}'
  character(len=*), parameter :: transfile = '{transfile}'

  character(len=*), parameter :: initfile = ''
  character(len=*), parameter :: initwt = ''

  character(len=*), parameter :: varfile = ''

  integer, parameter :: neq = {neq}, neps = {neps}, nobs = {nobs}, npara = {npara}, ny = {ny}
  integer, parameter :: t0 = 0
   
  real(wp),  parameter :: convergence_threshold = 1e-10

  ! data
  real(wp) :: YY(ny, nobs)

  ! prior stuff
  integer :: pshape(npara), pmask(npara)
  real(wp) :: pmean(npara), pstdd(npara), pfix(npara)

  ! prtr
  real(wp) :: trspec(4, npara)

  real, parameter :: verysmall = 0.0_wp!000001_wp
  integer :: nunstab = 0
  integer :: zxz = 0 
  integer :: fixdiv = 1
  real(wp) :: stake = 1.00_wp

  !$OMP THREADPRIVATE(nunstab,stake,zxz,fixdiv)


contains

  include '/mq/home/m1eph00/code/fortran/base/meyer_gohde.f90'

  {get_data} 

  {get_obs}

  {meas_sub}
        
  {N_matrix}

  {OMEGA_matrix} 

  {zero_matrices}

  {j_matrices}

  {inf_matrices}

  {pmsv}

  subroutine read_in()
    integer :: i
  open(1, file=priorfile, status='old', action='read')
  do i = 1, npara
     read(1, *) pshape(i), pmean(i), pstdd(i), pmask(i), pfix(i)
  end do
  close(1)

  open(1, file=datafile, status='old', action='read')
  do i = 1, nobs
     read(1, *) YY(:,i)
  end do
  close(1)

  open(1, file=transfile, status='old', action='read')
  do i = 1, npara
     read(1, *) trspec(:,i)
  end do
  close(1)
  end subroutine read_in

end module {name}
"""

        mod_str = fortran_template.format(**file_dict)

        mfile = open(os.path.join(mdir, name+".f90"), 'w')        
        mfile.write(mod_str)
        mfile.close()

    def __write_matlab_file(self, fstr):
        def print_matlab(x):
            return x.__str__().replace('**', '^').replace('oo', 'inf')

        ofile = open(fstr, 'w')
        
        if self.A is None:
            self.construct_sys_mat()

        ofile.write( "%------------------------------------------------------------\n")
        ofile.write( "%\n")
        ofile.write( "% Automatically generated m-file\n")
        ofile.write( "%\n")
        ofile.write( "%------------------------------------------------------------\n")

        #print "addpath('/mq/home/DSGE/research/alt_pricing/code/MAsolve/linlagex/')"
        parai = 1
        for p in self.parameters:
            ofile.write("{0} = p({1});\n".format(p.name, parai))
            parai += 1

        if self['para_func'] is not None:
            ofile.write("% Helper parameters\n")
            for v in self['other_para']:
                ofile.write("{0} = {1};\n".format(v.name, print_matlab(self['para_func'][v.name])))

        enames = ";".join(["\'" + x.__str__() + "\'" for x in self.endo_variables])
        ofile.write("ENDOGENOUS_VARIABLE_NAMES = {" + enames + "};\n")
        xnames = ";".join(["\'" + x.__str__() + "\'" for x in self.exo_variables])
        ofile.write("EXOGENOUS_VARIABLE_NAMES = {" + xnames + "};\n")

        ofile.write("it_max_value=\'infinity\';\n");
        ofile.write('max_lead_observable = ' + str(self.max_lead_observable) + ';');
        mat_list = dict()
        mat_list['A_0'] = self.A
        mat_list['B_0'] = self.B
        mat_list['C_0'] = self.C
        mat_list['F_0'] = self.F
        mat_list['G_0'] = self.G

        mat_list['N'] = self.N
        mat_list['Omega'] = self.Q

        mat_list['A_inf'] = self.Ainf
        mat_list['B_inf'] = self.Binf
        mat_list['C_inf'] = self.Cinf
        mat_list['F_inf'] = self.Finf
        mat_list['G_inf'] = self.Ginf
        mat_list['Q{1}'] = self.Q1
        mat_list['Q{2}'] = self.Q2
        mat_list['DD'] = self.DD
        
        for i in np.arange(1, 100):
            mat_list['Qlead{' + str(i) + '}'] = self.Qlead[i-1]


        for key in mat_list:
            defstr = "{0} = zeros({1}, {2});\n".format(key, mat_list[key].rows, mat_list[key].cols)
            ofile.write(defstr);
            for i in np.arange(mat_list[key].rows):
                for j in np.arange(mat_list[key].cols):
                    if mat_list[key][i, j] != 0:
                        pstr = "{0}({1:2d}, {2:2d}) = {3};\n".format(key, i+1, j+1, print_matlab(mat_list[key][i, j]))
                        ofile.write(pstr)
                    




        fun_list = dict()
        fun_list['A_j'] = self.Aj
        fun_list['B_j'] = self.Bj
        fun_list['C_j'] = self.Cj
        fun_list['F_j'] = self.Fj
        fun_list['G_j'] = self.Gj

        for key in fun_list:
            pstr = "{0} = @({1}) [".format(key, self.index)
            for i in np.arange(fun_list[key].rows):
                for j in np.arange(fun_list[key].cols):
                    pstr += " {0} ".format(print_matlab(fun_list[key][i, j]))

                if i < fun_list[key].rows - 1:
                    pstr += ";"

            pstr += '];\n'
            ofile.write(pstr);


        demean_str = """
        if exist('yy')
           yy = yy - repmat(DD', size(yy, 1), 1);
        end 
"""
        ofile.write(demean_str)
        ofile.close()
        #print "linlagex;"
