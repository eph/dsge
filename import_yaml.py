import yaml

from helper_functions import EXP, E, ASUM
from symbols import Variable, Parameter, Shock, Equation
from dolo.symbolic.symbolic import timeshift as TS

def import_yaml(fstr):
    f = open(fstr)
    txt = f.read()
    f.close()

    txt = txt.replace('^','**')
    txt = txt.replace(';','')

    model_yaml = yaml.load(txt)


    dec = model_yaml['declarations']

    name = dec['name']

    
