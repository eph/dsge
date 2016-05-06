from dsge.DSGE import DSGE

import pkg_resources

# nkmp
relative_loc = 'examples/nkmp/'
model_file = pkg_resources.resource_filename('dsge', relative_loc+'nkmp.yaml')
data_file = pkg_resources.resource_filename('dsge', relative_loc+'us.txt')

nkmp = DSGE.read(model_file)
nkmp['__data__']['estimation']['data']['file'] = data_file

# sw
relative_loc = 'examples/sw/'
model_file = pkg_resources.resource_filename('dsge', relative_loc+'sw.yaml')
data_file = pkg_resources.resource_filename('dsge', relative_loc+'YY.txt')

sw = DSGE.read(model_file)
sw['__data__']['estimation']['data']['file'] = data_file
