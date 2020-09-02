import numpy as np
import pandas as p 

import warnings

def read_data_file(datafile, obs_names):

    if type(datafile) == dict:
        startdate = datafile['start']
        datafile = datafile['file']
    else:
        startdate = 0

    try:
        with open(datafile, 'r') as df:
            data = df.read()
            delim_dict = {}

            if data.find(',') > 0:
                delim_dict['delimiter'] = ','

            data = np.genfromtxt(datafile, missing_values='NaN', **delim_dict)
    except:
        warnings.warn("%s could not be opened." % datafile)
        data = np.nan * np.ones((100, len(obs_names)))
        startdate = 0

    if len(obs_names) > 1:
        data = p.DataFrame(data[:, :len(obs_names)], columns=list(
            map(lambda x: str(x), obs_names)))
    else:
        data = p.DataFrame(data, columns=list(
            map(lambda x: str(x), obs_names)))

    if startdate is not 0:
        nobs = data.shape[0]
        data.index = p.period_range(startdate, freq='Q', periods=nobs)

    return data
