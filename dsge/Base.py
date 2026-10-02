import numpy as np

from abc import ABC

from typing import List, Union, Callable
from .symbols import Parameter
from sympy import Expr, lambdify
from sympy import default_sort_key, topological_sort
import sympy 

from .contexts import numeric_context, function_context
from .safe_expression import parse_expression, prepare_expression, validate_identifier, declared_functions

class Base(dict, ABC):
    """Base class for DSGE model"""
    
    lambdify_auxiliary = None
    def __lambdify_auxiliary(self, sort: bool = False, context=numeric_context) -> Callable:
        if sort:
            raise NotImplementedError('Sorting not implemented yet')

        aux = []
        context_aux = {}

        for key, value in self['auxiliary_parameters'].items():
            value = parse_expression(value, {p.name: p for p in self['parameters'] + list(self['auxiliary_parameters'])})
            aux.append(value.subs(context_aux))
            context_aux[key] = aux[-1]

        lambdified = lambdify([self['parameters']], aux,
                              modules=context)

        return lambdified
        

    def lambdify(
        self,
        expr_or_matrix: Union[Expr, sympy.Matrix],
        with_auxiliary: bool = False,
        context: dict | None = None,
    ) -> Callable:
        if context is None:
            context = {}
        all_parameters = [self['parameters'] + list(self['auxiliary_parameters'].keys())]
        for parameter in all_parameters[0]:
            validate_identifier(parameter.name)
        parse_context = {p.name: p for p in all_parameters[0]}
        parse_context.update(function_context)
        external = self.get('__data__', {}).get('declarations', {}).get('external', {})
        parse_context.update(declared_functions(external.get('names', [])))
        expr_or_matrix = prepare_expression(expr_or_matrix, parse_context)
        expanded_numeric = lambdify(all_parameters, expr_or_matrix,
                                    modules=[{'ImmutableDenseMatrix': np.array, **context}, 'numpy', 'scipy'])
        if with_auxiliary:
            return expanded_numeric
        else:
            if self.lambdify_auxiliary is None:
                auxiliary = self.__lambdify_auxiliary(context=context)
                self.lambdify_auxiliary = auxiliary
            else:
                auxiliary = self.lambdify_auxiliary
                
            def wrap_f(f):
                return lambda x: f([*x, *auxiliary(x)])
            return wrap_f(expanded_numeric)

        
        


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambdify_auxiliary = None
        self.create_string_attributes(['parameters', 'auxiliary_parameters'])

    def create_string_attributes(self, attributes: List[str]):
        for attr_name in attributes:
            symbols = self[attr_name]
            if isinstance(symbols, dict):
                string_list = [str(symbol) for symbol in symbols.keys()]
            else:
                string_list = [str(symbol) for symbol in symbols]
            setattr(self, f'{attr_name}', string_list)

    def fix_parameters(self, **kwargs):
        # Parse before mutating the model, so rejected input leaves it intact.
        parameters = list(self['parameters'])
        auxiliary = dict(self['auxiliary_parameters'])
        for name, value in kwargs.items():
            parameter = Parameter(name)
            parameters.remove(parameter)
            auxiliary[parameter] = value
        context = {p.name: p for p in parameters + list(auxiliary)}
        context.update(function_context)
        external = self.get('__data__', {}).get('declarations', {}).get('external', {})
        context.update(declared_functions(external.get('names', [])))
        parsed = {p: parse_expression(value, context) for p, value in auxiliary.items()}
        edges = [(p, dep) for p, value in parsed.items()
                 for dep in value.atoms(Parameter) if dep in parsed]
        order = topological_sort([list(parsed), edges], default_sort_key)[::-1]
        self['parameters'] = parameters
        self['auxiliary_parameters'] = {p: parsed[p] for p in order}
        self.lambdify_auxiliary = None
        self.create_string_attributes(['parameters', 'auxiliary_parameters'])
        return self

    def p0(self):
        return list(map(lambda x: self["calibration"]['parameters'][str(x)], self['parameters']))

    def update_data_file(self, file_path, start_date=None):
        if start_date is None:
            self['estimation']['data'] = file_path
        else:
            self['estimation']['data'] = {'file': file_path, 'start': start_date}

        return None


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

     
