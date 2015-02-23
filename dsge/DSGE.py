from __future__ import division

import sympy
import yaml
from symbols import Variable, Equation, Shock, Parameter, TSymbol
import sys
sys.path.append('/mq/home/m1eph00/python/dolo-master/')
from dolo.symbolic.symbolic import timeshift as TS

import scipy.stats as stats
from sympy.matrices import Matrix, zeros
import re
import numpy as np
import itertools
import pandas as p
from StateSpaceModel import LinearDSGEModel
class DSGE(dict):

    max_lead = 1;
    max_lag  = 1;


    def __init__(self, *kargs, **kwargs):
        super(DSGE, self).__init__(self, *kargs, **kwargs)

        fvars = []
        lvars = []



        # get forward looking variables
        for eq in self['equations']:

            variable_too_far = [v for v in eq.atoms() if isinstance(v, Variable) and v.date > 1]
            variable_too_early = [v for v in eq.atoms() if isinstance(v, Variable) and v.date < -1]


            eq_fvars = [v for v in eq.atoms() if isinstance(v, TSymbol) and v.date > 0]
            eq_lvars = [v for v in eq.atoms() if isinstance(v, TSymbol) and v.date < 0]


            fvars = list(set(fvars).union(eq_fvars))
            lvars = list(set(lvars).union(set(eq_lvars)))

        self['info']['nstate'] = len(self.variables) + len(fvars)

        self['fvars'] = fvars
        self['fvars_lagged'] = [Parameter('__LAGGED_'+f.name) for f in fvars]
        self['lvars'] = lvars
        self['re_errors'] = [Shock('eta_'+v.name) for v in self['fvars']]

        self['re_errors_eq'] = []
        i = 0
        for fv, lag_fv in zip(fvars, self['fvars_lagged']):
            self['re_errors_eq'].append(Equation(fv(-1) - lag_fv -  self['re_errors'][i], 0))
            i += 1

        if 'make_log' in self.keys():
            self['perturb_eq'] = []
            sub_dict = dict()
            sub_dict.update({v:Variable(v.name+'ss')*sympy.exp(v) for v in self['make_log']})
            sub_dict.update({v(-1):Variable(v.name+'ss')*sympy.exp(v(-1)) for v in self['make_log']})
            sub_dict.update({v(1):Variable(v.name+'ss')*sympy.exp(v(1)) for v in self['make_log']})

            for eq in self.equations:
                peq = eq.subs(sub_dict)
                self['perturb_eq'].append(peq)

            self['ss_ordering'] = [Variable(v.name+'ss') for v in self['make_log']]


        else:
            self['perturb_eq'] = self['equations']

        return

    def __repr__(self):
        return "A DSGE Model."

    @property
    def equations(self):
        return self['equations']

    @property
    def variables(self):
        return self['var_ordering']

    @property
    def parameters(self):
        return self['par_ordering']

    @property
    def shocks(self):
        return self['shk_ordering']

    @property
    def name(self):
        return self['name']

    @property
    def neq(self):
        return len(self['perturb_eq'])

    @property
    def neq_fort(self):
        return self.neq+self.neta

    @property
    def neta(self):
        return len(self['fvars'])

    @property
    def ns(self):
        return

    @property
    def ny(self):
        return len(self['observables'])

    @property
    def neps(self):
        return len(self['shk_ordering'])

    @property
    def npara(self):
        return len(self.parameters)

    def p0(self):
        return map(lambda x: self['calibration'][str(x)], self.parameters)

    def python_sims_matrices(self):

        from sympy.utilities.lambdify import lambdify
        vlist = self['var_ordering'] + self['fvars']
        llist = [l(-1) for l in self['var_ordering']] + self['fvars_lagged']
        slist = self['shk_ordering']

        subs_dict = dict()

        eq_cond = self['perturb_eq'] + self['re_errors_eq']

        subs_dict.update( {v:0 for v in self.variables})
        subs_dict.update( {v(-1):0 for v in self.variables})
        subs_dict.update( {v(1):0 for v in self.variables})

        svar = len(vlist)
        evar = len(slist)
        rvar = len(self['re_errors'])
        ovar = len(self['observables'])

        GAM0 = zeros((svar, svar))
        GAM1 = zeros((svar, svar))
        PSI  = zeros((svar, evar))
        PPI  = zeros((svar, rvar))

        eq_i = 0
        for eq in eq_cond:
            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))

            for v in curr_var:
                v_j = vlist.index(v)
                GAM0[eq_i, v_j] = -(eq).set_eq_zero.diff(v).subs(subs_dict)

            past_var = filter(lambda x: x in llist, eq.atoms())

            for v in past_var:
                v_j = llist.index(v)
                GAM1[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict)

            shocks = filter(lambda x: x, eq.atoms(Shock))

            for s in shocks:
                if s not in self['re_errors']:
                    s_j = slist.index(s)
                    PSI[eq_i, s_j] = eq.set_eq_zero.diff(s).subs(subs_dict)
                else:
                    s_j = self['re_errors'].index(s)
                    PPI[eq_i, s_j] = eq.set_eq_zero.diff(s).subs(subs_dict)

            eq_i += 1
            print "\r Differentiating equation {0} of {1}.".format(eq_i, len(eq_cond)),
        DD = zeros((ovar, 1))
        ZZ = zeros((ovar, svar))

        eq_i = 0
        for obs in self['observables']:
            eq = self['obs_equations'][str(obs)]

            DD[eq_i, 0] = eq.subs(subs_dict)

            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))
            for v in curr_var:
                v_j = vlist.index(v)
                ZZ[eq_i, v_j] = eq.diff(v).subs(subs_dict)

            eq_i += 1


        print ""
        from collections import OrderedDict
        subs_dict = []
        context = dict([(p.name, p) for p in self.parameters])
        context['exp'] = sympy.exp
        context['log'] = sympy.log
        import sys
        sys.stdout.flush()
        ss = {}

        for p in self['other_para']:
            ss[str(p)] = eval(str(self['para_func'][p.name]), context)
            context[str(p)] = ss[str(p)]
            print "\r Constructing substitution dictionary [{0:20s}]".format(p.name),
        sys.stdout.flush()
        print ""

        GAM0 = lambdify(self.parameters+self['other_para'], GAM0)
        GAM1 = lambdify(self.parameters+self['other_para'], GAM1)
        PSI = lambdify(self.parameters+self['other_para'], PSI)
        PPI = lambdify(self.parameters+self['other_para'], PPI)

        psi = lambdify(self.parameters, [ss[str(px)] for px in self['other_para']])
        def add_para_func(f):
            def wrapped_f(px):
                return f(*np.append(px, psi(*px)))
            return wrapped_f

        self.GAM0 = add_para_func(GAM0)
        self.GAM1 = add_para_func(GAM1)
        self.PSI = add_para_func(PSI)
        self.PPI = add_para_func(PPI)

        QQ = self['covariance'].subs(subs_dict)
        HH = self['measurement_errors'].subs(subs_dict)

        DD = DD.subs(subs_dict)
        ZZ = ZZ.subs(subs_dict)

        QQ = lambdify(self.parameters+self['other_para'], self['covariance'])
        HH = lambdify(self.parameters+self['other_para'], self['measurement_errors'])
        DD = lambdify(self.parameters+self['other_para'], DD)
        ZZ = lambdify(self.parameters+self['other_para'], ZZ)

        self.QQ = add_para_func(QQ)
        self.DD = add_para_func(DD)
        self.ZZ = add_para_func(ZZ)
        self.HH = add_para_func(HH)

        return GAM0, GAM1, PSI, PPI

    def compile_model(self):
        self.python_sims_matrices()

        GAM0 = self.GAM0
        GAM1 = self.GAM1
        PSI = self.PSI
        PPI = self.PPI

        QQ = self.QQ
        DD = self.DD
        ZZ = self.ZZ
        HH = self.HH


        datafile = self['__data__']['estimation']['data']

        if type(datafile)==dict:
            startdate = datafile['start']
            datafile = datafile['file']
        else:
            startdate = 0


        with open(datafile, 'r') as df:
            data = df.read()
            delim_dict = {}

            if data.find(',') > 0:
                delim_dict['delimiter'] = ','

            data = np.genfromtxt(datafile, missing_values='NaN', **delim_dict)

        if len(self['observables']) > 1:
            data = p.DataFrame(data[:, :len(self['observables'])], columns=map(lambda x: str(x), self['observables']))
        else:
            data = p.DataFrame(data, columns=map(lambda x: str(x), self['observables']))

        if startdate is not 0:
            nobs = data.shape[0]
            data.index = p.period_range(startdate, freq='Q', periods=nobs)

        prior = None
        if 'prior' in self['__data__']['estimation']:
            prior_type = ['beta', 'gamma', 'normal', 'inv_gamma', 'uniform', 'fixed']
            prior = []
            for par in self.parameters:
                prior_spec = self['__data__']['estimation']['prior'][par.name]

                ptype = prior_spec[0]
                pmean = prior_spec[1]
                pstdd = prior_spec[2]
                from scipy.stats import beta, norm, uniform, gamma
                from OtherPriors import InvGamma
                if ptype=='beta':
                    a = (1-pmean)*pmean**2/pstdd**2 - pmean
                    b = a*(1/pmean - 1)
                    prior.append(beta(a, b))
                if ptype=='gamma':
                    b = pstdd**2/pmean
                    a = pmean/b
                    prior.append(gamma(a, scale=b))
                if ptype=='normal':
                    a = pmean
                    b = pstdd
                    prior.append(norm(loc=a, scale=b))
                if ptype=='inv_gamma':
                    a = pmean
                    b = pstdd
                    prior.append(InvGamma(a, b))
                if ptype=='uniform':
                    a = pmean
                    b = pstdd
                    prior.append(uniform(loc=a, scale=b))
                    
        from Prior import Prior as pri
        dsge = LinearDSGEModel(data, GAM0, GAM1, PSI, PPI,
                               QQ, DD, ZZ, HH, t0=0,
                               shock_names=map(lambda x: str(x), self.shocks),
                               state_names=map(lambda x: str(x), self.variables+self['fvars']),
                               obs_names=map(lambda x: str(x), self['observables']),
                               prior=pri(prior))

        return dsge

    def solve_model(self, p0):

        #if self.GAM0 == None:
        self.python_sims_matrices()

        from gensys import gensys_wrapper as gensys

        G0 = self.GAM0(*p0)
        G1 = self.GAM1(*p0)
        PSI = self.PSI(*p0)
        PPI = self.PPI(*p0)
        C0 = np.zeros((G0.shape[0]))
        TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensys.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)

        return TT, RR, RC

    def sims_matrices(self):

        vlist = self['var_ordering'] + self['fvars']
        llist = self['lvars']
        slist = self['shk_ordering']

        subs_dict = dict()

        subs_dict.update( {v:0 for v in self.variables})
        subs_dict.update( {v(-1):0 for v in self.variables})
        subs_dict.update( {v(1):0 for v in self.variables})

        from extension.fcode import fcode
        def wrapper_fortran(expr):
            return fcode(expr.subs([(si, sympy.Float(si)) for si in expr.atoms(sympy.Integer)]))


        wf = wrapper_fortran
        my_str = ''

        eqi = 1
        str_list = []
        for peq in self['perturb_eq']:
            #print "    !------------------------------------------------------------"
            #print ("    ! Equation {0}".format(str(eqi)))
            #print "    !------------------------------------------------------------"
            for v in filter(lambda x: x.date >= 0, peq.atoms(Variable)):

                D = peq.set_eq_zero.diff(v)

                derivative = D.subs(subs_dict)

                if not(derivative==0):
                    #my_str = my_str + '    GAM0({0}, {1}) = {2}\n'.format(str(eqi), v.fortind, wf(-derivative))
                    str_list.append('    GAM0({0}, {1}) = {2}\n'.format(str(eqi), v.fortind, wf(-derivative)))

            for v in llist:
                derivative = peq.set_eq_zero.diff(v).subs(subs_dict)

                if not(derivative==0):
                    #my_str = my_str + '    GAM1({0}, {1}) = {2}\n'.format(str(eqi), v.fortind, wf(derivative))
                    str_list.append('    GAM1({0}, {1}) = {2}\n'.format(str(eqi), v.fortind, wf(derivative)))

            for v in filter(lambda x: x not in self['re_errors'], peq.atoms(Shock)):
                derivative = peq.set_eq_zero.diff(v).subs(subs_dict)

                if not(derivative==0):
                    #my_str = my_str + '    PSI({0}, {1}) = {2}\n'.format(str(eqi), v.fortind, wf(derivative))
                    str_list.append('    PSI({0}, {1}) = {2}\n'.format(str(eqi), v.fortind, wf(derivative)))
            eqi = eqi + 1

            str_list.append('\n\n\n')
            print "\r Parsing equation %d" % eqi,

        eti = 1
        for fv in self['fvars']:
            str_list.append('     GAM0({n}, {v}) = 1.0_wp\n     GAM1({n}, {Ev}) = 1.0_wp\n     PPI({n}, {i}) = 1.0_wp\n\n\n'.format(n=eqi, v=fv(-1).fortind, Ev=fv.fortind, i=eti))
            eti += 1
            eqi += 1



        return ''.join(str_list)

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


        if 'estimation' in self['__data__']:
            datafile = self['__data__']['estimation']['data']
            data = np.genfromtxt(datafile, delimiter=',', missing_values='NaN')

            datafile = os.path.join(odir, "yy.txt")
            np.savetxt(datafile, data)




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

    def __write_matlab_file(self, fstr):

        vlist = self['var_ordering'] + self['fvars']
        llist = self['lvars']
        slist = self['shk_ordering']

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
        for i in range(0, self['covariance'].shape[0]):
            for j in range(0, self['covariance'].shape[1]):

                if not(self['covariance'][i, j]==0):
                    stri = self['shk_ordering'][i].fortind
                    strj = self['shk_ordering'][i].fortind
                    QQstr = QQstr + '    QQ({0}, {1}) = {2};\n'.format(stri, strj, self['covariance'][i, j])

        ZZstr = ''
        DDstr = ''

        all_vars_in_obs = [self['obs_equations'][key].atoms(Variable) for key in self['obs_equations']]
        max_lead_in_obs = max([i.date for i in itertools.chain.from_iterable(all_vars_in_obs)])

        subsdict = dict.fromkeys([x for x in itertools.chain.from_iterable(all_vars_in_obs)], 0)
        obsi = 1
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


        if max_lead_in_obs > 0:
            ZZstr = ZZstr + "\n\n\n TTn = TT\n"

        for lead_obs in np.arange(1, max_lead_in_obs+1):

            obsi = 1
            for o in self['observables']:
                ee = self['obs_equations'][o.name]
                vi = 1

                for v in self.variables:
                    z = ee.diff(v(lead_obs))

                    if not(z==0):
                        ZZstr = ZZstr + ' ZZ({0}, :) = ZZ({0}, :) + {2} * TTn({1}, :);\n'.format(obsi, vi, z)

                    vi = vi + 1


                obsi = obsi + 1


            ZZstr = ZZstr + 'TTn = TT*TTn;\n'


        gensys_file = """
function [TT, RR, QQ, DD, ZZ, HH, RC] = {mname}_model(para)
{para}
{dec}
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
        var_string = ''
        nvars = len(self['var_ordering'])
        for i in xrange(1, nvars+1):
            var_string = var_string + "    {0} = {1};\n".format(self['var_ordering'][i-1].fortind, i)
            #print var_string

        nevars = len(self['fvars'])
        for i in xrange(nvars+1, nvars+nevars+1):
            var_string = var_string + "    {0} = {1};\n".format(self['fvars'][i-nvars-1].fortind, i)


        for i in xrange(1, len(self['shk_ordering'])+1):
            var_string = var_string + "    {0} = {1};\n".format(self['shk_ordering'][i-1].fortind, i)

        ss_mat = QQstr + ZZstr + DDstr;
        w = gensys_file.format(mname=self.name, dec=var_string, para=assign_list, gensys_mat=my_str, obs_mat=ss_mat)
        w = w.replace('**', '^')
        fout = open(fstr, 'w')
        fout.write(w)
        fout.close()


    def write_fortran_module(self, odir, name=None):

        #basedir = '/mq/home/m1eph00/code/models/{0}/'.format(directory)

        if name==None:
            name = self.name

        import os

        if not os.path.isabs(odir):
            odir = os.path.join(os.getcwd(), odir)

        print "I\'m writing to ", odir, "."

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
        smcfile = open('/mq/home/m1eph00/python-repo/dsge/templates/smc_driver_mpi.f90', 'r')
        smc_driver = smcfile.read()
        smcfile.close()

        smcfilename = os.path.join(odir, 'smc_driver_mpi.f90')
        smcfile = open(smcfilename, 'w')
        smcfile.write(smc_driver.format(**{'model': name}))
        smcfile.close()

        #------------------------------------------------------------
        # RWMH
        #------------------------------------------------------------
        rwmhfile = open('/mq/home/m1eph00/python-repo/dsge/templates/rwmh_driver.f90', 'r')
        rwmh_driver = rwmhfile.read()
        rwmhfile.close()

        rwmhfilename = os.path.join(odir, 'rwmh_driver.f90')
        rwmhfile = open(rwmhfilename, 'w')
        rwmhfile.write(rwmh_driver.format(**{'model': name}))
        rwmhfile.close()

        #------------------------------------------------------------
        # Block MCMC
        #------------------------------------------------------------
        rwmhfile = open('/mq/home/m1eph00/python-repo/dsge/templates/blockmcmc.f90', 'r')
        rwmh_driver = rwmhfile.read()
        rwmhfile.close()

        rwmhfilename = os.path.join(odir, 'blockmcmc.f90')
        rwmhfile = open(rwmhfilename, 'w')
        rwmhfile.write(rwmh_driver.format(**{'model': name}))
        rwmhfile.close()



        #------------------------------------------------------------
        # MAKEFILE
        #------------------------------------------------------------
        makefile = open('/mq/home/m1eph00/python-repo/dsge/templates/Makefile_dsge', 'r')
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
        # now, the data
        #------------------------------------------------------------
        datafile = self['__data__']['estimation']['data']

        with open(datafile, 'r') as df:
            data = df.read()
            delim_dict = {}

            if data.find(',') > 0:
                delim_dict['delimiter'] = ','

            data = np.genfromtxt(datafile, missing_values='NaN', **delim_dict)

        if len(data.shape) > 1:
            nobs, ny = data.shape
        else:
            nobs = data.shape[0]
            ny = 1

        # if not ny == self.nobs_vars:
        #     print "ERROR SECTOR 12"

        datafile = os.path.join(mdir, "yy.txt")
        np.savetxt(datafile, data)


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
                if prior_spec[0]=='uniform':
                    trspec['uniform'] = [1, prior_spec[1], prior_spec[2], 1]

                tfile.write("{0}, {1}, {2}, {3}\n".format(*trspec[prior_spec[0]]))
            tfile.close()



        file_dict = dict()
        from extension.fcode import fcode

        e = re.compile(r'([^a-zA-Z][0-9\.]+)')
        def wrapper_fortran(expr):
            fstr = fcode(expr.subs([(si, sympy.Float(si)) for si in expr.atoms(sympy.Integer)]))
            #fstr = e.sub(r'\1_wp', fstr)
            # hack
            return fstr


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



        file_dict['name'] = name
        file_dict['sims_mat'] = self.sims_matrices()


        file_dict['pfile'] = mdir + '/prior.txt'
        file_dict['datafile'] = mdir + '/yy.txt'
        file_dict['transfile'] = mdir + '/trans.txt'

        file_dict['npara'] = self.npara

        cal_list = [str(self['calibration'][str(x)]) for x in self.parameters]
        cal_list = [re.sub(r'([^a-zA-Z][0-9\.]+)', r'\1_wp',str_p) for str_p in cal_list]
        pmsv = 'function pmsv result(para)\n  real(wp) :: para(npara)\n\n'
        pmsv += "para = (/" + ",".join(cal_list) + "/)\n\n end function pmsv"

        file_dict['pmsv'] = pmsv


        dec = (#"\n    ! deep parameters \n"+
               #"    real(wp) :: {0}".format(", ".join([x.__str__() for x in self['par_ordering']])) +
               # "\n    ! not so deep parameters \n"+
               # "    real(wp) :: {0}".format(", ".join([x.__str__() for x in self['other_parameters']]))+
               # "\n    ! steady states \n"+
               # "    real(wp) :: {0}".format(", ".join([x.__str__() for x in self['ss_ordering']]))+
               "\n    ! variables \n"+
               "    integer :: {0} \n".format(", ".join([x.fortind for x in self.variables + self['fvars']]))+
               "\n    ! shocks \n"+
               "    integer :: {0}".format(", ".join([x.fortind for x in self['shk_ordering']])))

        file_dict['dec'] = dec

        file_dict['neq'] = self.neq_fort
        file_dict['neta'] = self.neta
        file_dict['neps'] = self.neps
        file_dict['nobs'] = nobs
        file_dict['npara'] = self.npara
        file_dict['ny'] = self.ny
        file_dict['t0'] = 0
        para_list = ",".join([p.__str__() for p in self.parameters])
        helper_list = ",".join([p.__str__() for p in self['para_func']])

        file_dict['para_list'] = para_list
        file_dict['helper_list'] = helper_list
        file_dict['assign_list'] = assign_list
        var_string = ''

        nvars = len(self.variables)
        for i in xrange(1, nvars+1):
            var_string = var_string + "    {0} = {1}\n".format(self['var_ordering'][i-1].fortind, i)
            #print var_string

        nevars = len(self['fvars'])
        for i in xrange(nvars+1, nvars+nevars+1):
            var_string = var_string + "    {0} = {1}\n".format(self['fvars'][i-nvars-1].fortind, i)


        for i in xrange(1, self['info']['nshock']+1):
            var_string = var_string + "    {0} = {1}\n".format(self['shk_ordering'][i-1].fortind, i)
        file_dict['def'] = var_string
        file_dict['para'] = assign_list

        QQstr = ''

        for i in range(0, self['covariance'].shape[0]):
            for j in range(0, self['covariance'].shape[1]):

                if not(self['covariance'][i, j]==0):
                    stri = self['shk_ordering'][i].fortind
                    strj = self['shk_ordering'][i].fortind
                    QQstr = QQstr + '    QQ({0}, {1}) = {2}\n'.format(stri, strj, self['covariance'][i, j])

        HHstr = ''
        for i in range(0, self['measurement_errors'].shape[0]):
            for j in range(0, self['measurement_errors'].shape[1]):

                if not(self['measurement_errors'][i, j]==0):
                    #stri = self['measurement_errors'][i].fortind
                    #strj = self['measurement_errors'][i].fortind
                    HHstr = HHstr + '    HH({0}, {1}) = {2}\n'.format(i+1, j+1, self['measurement_errors'][i, j])


        ZZstr = ''
        DDstr = ''
        obsi = 1
        subsdict = dict(zip(self.variables, np.zeros(nvars)))




        all_vars_in_obs = [self['obs_equations'][key].atoms(Variable) for key in self['obs_equations']]
        max_lead_in_obs = max([i.date for i in itertools.chain.from_iterable(all_vars_in_obs)])


        subsdict = dict.fromkeys([x for x in itertools.chain.from_iterable(all_vars_in_obs)], 0)
        for o in self['observables']:
            ee = self['obs_equations'][o.name]
            DDm = ee.subs(subsdict)
            DDstr = DDstr + 'DD({0}) = {1};\n'.format(obsi, wrapper_fortran(DDm))

            vi = 1
            for v in self.variables:
                z = ee.diff(v)
                if not(z==0):
                    ZZstr = ZZstr + '   ZZ({0}, {1}) = {2};\n'.format(obsi, vi, wrapper_fortran(z))

                vi = vi+1

            obsi = obsi+1


        if max_lead_in_obs > 0:
            ZZstr = ZZstr + "\n\n\n TTn = TT;\n"

        for lead_obs in np.arange(1, max_lead_in_obs+1):

            obsi = 1
            for o in self['observables']:
                ee = self['obs_equations'][o.name]
                vi = 1

                for v in self.variables:
                    z = ee.diff(v(lead_obs))

                    if not(z==0):
                        ZZstr = ZZstr + ' ZZ({0}, :) = ZZ({0}, :) + {2} * TTn({1}, :);\n'.format(obsi, vi, wrapper_fortran(z))

                    vi = vi + 1


                obsi = obsi + 1


            ZZstr = ZZstr + 'call dgemm(\'n\', \'n\', neq, neq, neq, 1.0_wp, TT, neq, TTn, neq, 0.0_wp, temp, neq)\n TTn = temp\n\n'

        file_dict['ss_mat'] = QQstr + ZZstr + DDstr + HHstr

        if 'helper_func' in self['__data__']['declarations']:
            extra_includes = 'include \'' + self['__data__']['declarations']['helper_func']['fortran'] + '\''
        else:
            extra_includes = ''

        file_dict['extra_includes'] = extra_includes

        fortran_template = """
!------------------------------------------------------------
! Automatically generated fortran file.
!------------------------------------------------------------
module {name}
  use mkl95_precision, only: wp => dp

  use prior
  use filter
  use particle_filter
        
  use gensys


  implicit none

  character(len=*), parameter :: mname = '{name}'

  character(len=*), parameter :: priorfile = '{pfile}'
  character(len=*), parameter :: priorfile2 = ''
  character(len=*), parameter :: datafile = '{datafile}'
  character(len=*), parameter :: transfile = '{transfile}'

  character(len=*), parameter :: initfile = ''
  character(len=*), parameter :: initwt = ''

  character(len=*), parameter :: varfile = ''

  integer, parameter :: neq = {neq}, neta = {neta}, neps = {neps}, nobs = {nobs}, npara = {npara}, ns = {neq}, ny = {ny}
  integer, parameter :: t0 = {t0}

  real(wp),  parameter :: REALLY_NEG = -1000000000000.0_wp
  ! data
  real(wp) :: YY(ny, nobs)

  ! prior stuff
  integer :: pshape(npara), pmask(npara)
  real(wp) :: pmean(npara), pstdd(npara), pfix(npara)

  integer :: pshape2(npara), pmask2(npara)
  real(wp) :: pmean2(npara), pstdd2(npara), pfix2(npara)

  ! prtr
  real(wp) :: trspec(4, npara)

contains

  {extra_includes}
  include '/mq/home/m1eph00/code/fortran/base/helper_functions.f90'

  subroutine sysmat(para, TT, RR, QQ, DD, ZZ, HH, info)

    real(wp), intent(in) :: para(npara)

    real(wp), intent(out) :: TT(neq, neq), RR(neq, neps), QQ(neps, neps), DD(ny), ZZ(ny, neq), HH(ny,ny)
    integer, intent(out) :: info

    real(wp) :: GAM0(neq, neq), GAM1(neq, neq), C(neq), PSI(neq, neps), PPI(neq, neta), CC(neq)
    real(wp) :: TTn(neq, neq), temp(neq, neq)

    real(wp) :: {para_list}
    real(wp) :: {helper_list}

    ! gensys
    real(wp) :: fmat, fwt, ywt, gev, loose, DIV
    integer :: eu(2)

    {dec}

    {para}

    {def}

    GAM0 = 0.0_wp
    GAM1 = 0.0_wp
    PSI = 0.0_wp
    PPI = 0.0_wp
    C = 0.0_wp

    {sims_mat}

    call do_gensys(TT, CC, RR, fmat, fwt, ywt, gev, eu, loose, GAM0, GAM1, C, PSI, PPI, DIV)
!
    info = eu(1)*eu(2)

    QQ = 0.0_wp
    ZZ = 0.0_wp
    DD = 0.0_wp
    HH = 0.0_wp
    {ss_mat}

    return
  end subroutine sysmat

  {pmsv}

end module {name}
"""

        mod_str = fortran_template.format(**file_dict)

        mod_str = mod_str.replace('_wp_wp', '_wp')
        mfile = open(os.path.join(mdir, name+".f90"), 'w')

        mfile.write(mod_str)
        mfile.close()

        # print("model {0}".format(self.name))

        # model_str = "  use mkl95_precision, only: wp => dp\n\n  use prior\n  use filter\n  use gensys\n\n  implicit none"
        # print model_str

        # char_str = "  character(len=*), parameter :: {0} = {1}{2}"

        # print(char_str.format('mname', self.name, ''))
        # print(char_str.format('priorfile', directory, 'prior.txt'))
        # print(char_str.format('datafile', directory, 'data.txt'))
        # print(char_str.format('transfile', directory, 'trspec.txt'))


        # print("contains")

        # print "  include /mq/home/m1eph00/code/fortran/base/helper_functions.f90"
        # print "  subroutine sysmat(para, TT, RR, QQ, DD, ZZ, HH, info)"
        # print "\n    integer, intent(out) :: info"
        # print "\n    real(wp), intent(in) :: para(npara)"
        # print "\n    real(wp), intent(out) :: TT(neq, neq), RR(neq, neps), QQ(neps, neps), DD(ny), ZZ(ny, neq), HH(ny, ny)"

        # print "\n    real(wp) :: GAM0(neq, neq), GAM1(neq, neq), C(neq), PSI(neq, neps), PPI(neq, neta), CC(neq), G0(neq, neq), G1(neq, neq)"

        # print "\n    ! deep parameters "
        # print("    real(wp) :: {0}".format(", ".join([x.__str__() for x in self['par_ordering']])))

        # print "\n    ! not so deep parameters"
        # print("    real(wp) :: {0}".format(", ".join([x.__str__() for x in self['other_parameters']])))

        # print "\n    ! steady states "
        # print("    real(wp) :: {0}".format(", ".join([x.__str__() for x in self['ss_ordering']])))

        # print "\n    ! variables "
        # print("    integer :: {0}".format(", ".join([x.fortind for x in self.variables + self['fvars']])))

        # print "\n    ! shocks "
        # print("    integer :: {0}".format(", ".join([x.fortind for x in self['shk_ordering']])))


        # nvars = len(self['var_ordering'])

        # for i in xrange(1, nvars+1):
        #     var_string = "    {0} = {1}".format(self['var_ordering'][i-1].fortind, i)
        #     print var_string

        # nevars = len(self['fvars'])
        # for i in xrange(nvars+1, nvars+nevars+1):
        #     var_string = "    {0} = {1}".format(self['fvars'][i-nvars-1].fortind, i)
        #     print var_string


        # for i in xrange(1, self['info']['nshock']+1):
        #     print("    {0} = {1}".format(self['shk_ordering'][i-1].fortind, i))

        # for i in xrange(1, len(self['fvars'])+1):
        #     print("    n_{0} = {1}".format(self['fvars'][i-1].name, i))



        # for i in range(0, self['covariance'].shape[0]):
        #     for j in range(0, self['covariance'].shape[1]):

        #         if not(self['covariance'][i, j]==0):
        #             stri = self['shk_ordering'][i].fortind
        #             strj = self['shk_ordering'][i].fortind
        #             print('    QQ({0}, {1}) = {2}'.format(stri, strj, self['covariance'][i, j]))




    @staticmethod
    def read(mfile):
        f = open(mfile)
        txt = f.read()
        f.close()


        txt = txt.replace('^', '**')
        txt = txt.replace(';', '')
        txt = re.sub(r"@ ?\n", " ", txt)
        model_yaml = yaml.load(txt)

        dec = model_yaml['declarations']
        cal = model_yaml['calibration']

        name = dec['name']

        var_ordering = [Variable(v) for v in dec['variables']]
        par_ordering = [Parameter(v) for v in dec['parameters']]
        shk_ordering = [Shock(v) for v in dec['shocks']]


        if 'para_func' in dec:
            other_para   = [Parameter(v) for v in dec['para_func']]
        else:
            other_para = []

        if 'observables' in dec:
            observables = [Variable(v) for v in dec['observables']]
            obs_equations = model_yaml['equations']['observables']
        else:
            observables = []
            obs_equations = dict()

        if 'measurement_errors' in dec:
            measurement_errors = [Shock(v) for v in dec['measurement_errors']]
        else:
            measurement_errors = None


        if 'make_log' in dec:
            make_log = [Variable(v) for v in dec['make_log']]
        else:
            make_log = []

        steady_state = [0]
        init_values = [0]

        context = [(s.name,s) for s in var_ordering + par_ordering + shk_ordering + other_para]
        context = dict(context)
        #context['TS'] = TS

        for f in [sympy.log, sympy.exp,
                  sympy.sin, sympy.cos, sympy.tan,
                  sympy.asin, sympy.acos, sympy.atan,
                  sympy.sinh, sympy.cosh, sympy.tanh,
                  sympy.pi, sympy.sign]:
            context[str(f)] = f

        context['sqrt'] = sympy.sqrt
        context['__builtins__'] = None

        equations = []

        raw_equations = model_yaml['equations']['model']


        for eq in raw_equations:
            if '=' in eq:
                lhs, rhs = str.split(eq, '=')
            else:
                lhs = teq
                rhs = '0'

            lhs = eval(lhs, context)
            rhs = eval(rhs, context)

            equations.append(Equation(lhs, rhs))



        #------------------------------------------------------------
        # Figure out max leads and lags
        #------------------------------------------------------------
        it = itertools.chain.from_iterable

        max_lead_exo = dict.fromkeys(shk_ordering)
        max_lag_exo = dict.fromkeys(shk_ordering)

        all_shocks = [list(eq.atoms(Shock)) for eq in equations]

        for s in shk_ordering:
            max_lead_exo[s] = max([i.date for i in it(all_shocks) if i.name == s.name])
            max_lag_exo[s] = min([i.date for i in it(all_shocks) if i.name == s.name])


        # arbitrary lags of exogenous shocks
        for s in shk_ordering:
            if abs(max_lag_exo[s]) > 0:
                var_s = Variable(s.name+"_VAR")
                var_ordering.append(var_s)
                equations.append(Equation(var_s, s))

                subs1 = [s(-i) for i in np.arange(1, abs(max_lag_exo[s])+1)]
                subs2 = [var_s(-i) for i in np.arange(1, abs(max_lag_exo[s])+1)]
                subs_dict = dict(zip(subs1, subs2))
                equations = [eq.subs(subs_dict) for eq in equations]


        all_vars = [list(eq.atoms(Variable)) for eq in equations]
        max_lead_endo = dict.fromkeys(var_ordering)
        max_lag_endo = dict.fromkeys(var_ordering)

        for v in var_ordering:
            max_lead_endo[v] = max([i.date for i in it(all_vars) if i.name == v.name])
            max_lag_endo[v] = min([i.date for i in it(all_vars) if i.name == v.name])

        #------------------------------------------------------------
        # arbitrary lags/leads of exogenous shocks
        subs_dict = dict()
        old_var = var_ordering[:]
        for v in old_var:

            # lags
            for i in np.arange(2, abs(max_lag_endo[v])+1):
                # for lag l need to add l-1 variable
                var_l = Variable(v.name + "_LAG" + str(i-1))

                if i == 2:
                    var_l_1 = Variable(v.name, date=-1)
                else:
                    var_l_1 = Variable(v.name + "_LAG" + str(i-2), date=-1)

                subs_dict[Variable(v.name, date=-i)] = var_l(-1)
                var_ordering.append(var_l)
                equations.append(Equation(var_l, var_l_1))


            # still need to do leads

        equations = [eq.subs(subs_dict) for eq in equations]

        cov = cal['covariances']

        nshock = len(shk_ordering)
        npara  = len(par_ordering)

        info = {'nshock': nshock, 'npara': npara}
        QQ = sympy.zeros((nshock, nshock))
        for key, value in cov.iteritems():
            shocks = key.split(",")

            if len(shocks)==1:
                shocks.append(shocks[0])

            if len(shocks)==2:
                shocki = Shock(shocks[0].strip())
                shockj = Shock(shocks[1].strip())

                indi = shk_ordering.index(shocki)
                indj = shk_ordering.index(shockj)

                QQ[indi, indj] = eval(value, context)
                QQ[indj, indi] = QQ[indi, indj]

            else:
                "fdfadsf"

        nobs = len(obs_equations)
        HH = sympy.zeros((nobs, nobs))

        if measurement_errors is not None:
            for key, value in cal['measurement_errors'].iteritems():
                shocks = key.split(",")

                if len(shocks)==1:
                    shocks.append(shocks[0])

                if len(shocks)==2:
                    shocki = Shock(shocks[0].strip())
                    shockj = Shock(shocks[1].strip())

                    indi = measurement_errors.index(shocki)
                    indj = measurement_errors.index(shockj)

                    HH[indi, indj] = eval(value, context)
                    HH[indj, indi] = HH[indi, indj]




        context['sum'] = np.sum
        context['range'] = range
        for obs in obs_equations.iteritems():
            obs_equations[obs[0]] = eval(obs[1], context)


        calibration = model_yaml['calibration']['parameters']

        model_dict = {
            'var_ordering': var_ordering,
            'par_ordering': par_ordering,
            'shk_ordering': shk_ordering,
            'other_parameters': other_para,
            'other_para': other_para,
            'para_func': cal['parafunc'],
            'calibration': calibration,
            'steady_state': steady_state,
            'init_values': init_values,
            'equations': equations,
            'covariance': QQ,
            'measurement_errors': HH,
            'meas_ordering': measurement_errors,
            'info': info,
            'make_log': make_log,
            '__data__': model_yaml,
            'name': dec['name'],
            'observables': observables,
            'obs_equations': obs_equations
            }


        model = DSGE(**model_dict)
        return model


def iteritems(d):
    return zip(d.keys(), d.values())
