from abc import ABC, abstractmethod

from typing import List, Dict, Tuple, Union, Callable, Optional
from .symbols import Parameter, Variable, Shock, Equation
from sympy import Expr
from sympy import default_sort_key, topological_sort
import sympy 

from .contexts import function_context 

class Base(dict, ABC):
    """Base class for DSGE model"""
    
    def parameters(self) -> List[str]:
        return [p.name for p in self['parameters']]

    def fixed_parameters(self) -> Dict[Parameter, Expr]:
        return self['auxiliary_parameters']

    def fix_parameters(self, **kwargs):
        for para, value in kwargs.items():
            para = Parameter(para)
            self['parameters'].remove(para)
            self['auxiliary_parameters'][para] = value

            context_tuple = [(p.name, p) for p in       self['parameters']+list(self['auxiliary_parameters'].keys())]


        context = dict(context_tuple)
        context.update(function_context)

        
        to_replace = [(p, sympy.sympify(value, context))
                      for p,value in self['auxiliary_parameters'].items()]


        edges = [(v[0],dep) for v in to_replace for dep in sympy.sympify(v[1]).atoms(Parameter) if dep in self['auxiliary_parameters'].keys()]

        para_func = topological_sort([self['auxiliary_parameters'].keys(), edges], default_sort_key)[::-1]
        # sort the fixed parameters in the order of their dependencies
        self['auxiliary_parameters'] = {p: sympy.sympify(self['auxiliary_parameters'][p], context) for p in para_func}
        return self

    def p0(self):
        return list(map(lambda x: self["calibration"]['parameters'][str(x)], self['parameters']))


    # only use create fortran model if fortress is available

    # def create_fortran_model(self, model_dir='__fortress_tmp', **kwargs):
    #     try:
    #         from fortress import make_smc
    #     except ImportError:
    #         raise ImportError('Fortress is not installed. Please install fortress to use this method.')
        
    #     smc_file = self.smc(self)
    #     model_linear = self.compile_model()

    #     other_files = {'data.txt': model_linear.yy,          
    #                    'prior.txt': 'prior.txt'}
    #     make_smc(smc_file, model_dir, other_files=other_files,**kwargs)                      
    #     write_prior_file(model_linear.prior, model_dir)           

# write a quick unittest for this
if __name__ == "__main__":
    parameters = [Parameter('beta'), Parameter('zeta'), Parameter('gamma'), Parameter('rho')]
    auxiliary_parameters = {Parameter('gamma'): 0.2, Parameter('kappa'): 0.1}
    test_model = Base({'parameters': parameters, 'auxiliary_parameters': auxiliary_parameters})
    test_model.fix_parameters(beta='0.99*gamma', gamma=0.5)
    print('Parameters', test_model['parameters'])
    print('Derived parameters', test_model['auxiliary_parameters'])

     
