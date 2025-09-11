from dsge import read_yaml
from dsge.resource_utils import resource_file_path

# Use persistent resource paths so downstream code can open later

# nkmp
_nkmp_model = resource_file_path('examples/nkmp/nkmp.yaml')
_nkmp_data = resource_file_path('examples/nkmp/us.txt')

nkmp = read_yaml(str(_nkmp_model))
nkmp['__data__']['estimation']['data']['file'] = str(_nkmp_data)

# sw
_sw_model = resource_file_path('examples/sw/sw.yaml')
_sw_data = resource_file_path('examples/sw/YY.txt')

sw = read_yaml(str(_sw_model))
sw['__data__']['estimation']['data']['file'] = str(_sw_data)
