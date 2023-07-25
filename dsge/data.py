import numpy as np
import pandas as pd

import warnings


def read_data_file(datafile, obs_names):
    '''
    Read data from a given file, and convert it to a pandas DataFrame.

    Parameters:
    datafile (str or dict): The path to the file containing the data, or a dictionary containing the path and start date.
        If the input is a dictionary, it must contain the following keys:
            "file": (str) The path to the file containing the data.
            "start": (str) The start date of the data in pandas Period format
    obs_names (list of str): A list of column names for the DataFrame.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the data from the input file.

    Raises:
    pandas.errors.ParserError: If the data in the input file cannot be parsed successfully.
    FileNotFoundError: If the input file does not exist in the file system.

    Example:
    >>> datafile = "/path/to/data.csv"
    >>> obs_names = ["A", "B", "C"]
    >>> data = read_data_file(datafile, obs_names)
    '''
    if type(datafile) == dict:
        startdate = datafile["start"]
        datafile = datafile["file"]
    else:
        startdate = 0

    try:
        # First, we'll try to load it with a header
        data = pd.read_csv(datafile, names=obs_names)
        # If the first row isn't numeric (excluding nan), we know it's the header.
        # If not, we load again without a header
        if data.iloc[0].apply(lambda x: pd.to_numeric(x, errors='coerce')).isna().any():
            data = pd.read_csv(datafile)
    except Exception as e:
        warnings.warn(f"{datafile} could not be opened. Error: {e}")
        data = pd.DataFrame(np.nan * np.ones((100, len(obs_names))), columns=obs_names)
        startdate = 0

    if startdate != 0:
        nobs = data.shape[0]
        # get either 'M or 'Q' from the startdate
        freq = 'Q' if 'Q' in startdate else 'M'
        data.index = pd.period_range(startdate, freq=freq, periods=nobs)

    return data
import numpy as np
#import pandas as p
#
#import warnings
#
#
#def read_data_file(datafile, obs_names):
#    '''
#    Read data from a given file, and convert it to a pandas DataFrame.
#
#    Parameters:
#    datafile (str or dict): The path to the file containing the data, or a dictionary containing the path and start date.
#        If the input is a dictionary, it must contain the following keys:
#            "file": (str) The path to the file containing the data.
#            "start": (str) The start date of the data in pandas Period format
#    obs_names (list of str): A list of column names for the DataFrame.
#
#    Returns:
#    pd.DataFrame: A pandas DataFrame containing the data from the input file.
#
#    Raises:
#    pandas.errors.ParserError: If the data in the input file cannot be parsed successfully.
#    FileNotFoundError: If the input file does not exist in the file system.
#
#    Example:
#    >>> datafile = "/path/to/data.csv"
#    >>> obs_names = ["A", "B", "C"]
#    >>> data = read_data_file(datafile, obs_names)
#    '''
#    if type(datafile) == dict:
#        startdate = datafile["start"]
#        datafile = datafile["file"]
#    else:
#        startdate = 0
#
#    try:
#        with open(datafile, "r") as df:
#            data = df.read()
#            delim_dict = {}
#
#            if data.find(",") > 0:
#                delim_dict["delimiter"] = ","
#
#            data = np.genfromtxt(datafile, missing_values="NaN", **delim_dict)
#    except:
#        warnings.warn("%s could not be opened." % datafile)
#        data = np.nan * np.ones((100, len(obs_names)))
#        startdate = 0
#
#    if len(obs_names) > 1:
#        data = p.DataFrame(
#            data[:, : len(obs_names)], columns=list(map(lambda x: str(x), obs_names))
#        )
#    else:
#        data = p.DataFrame(data, columns=list(map(lambda x: str(x), obs_names)))
#
#    if startdate != 0:
#        nobs = data.shape[0]
#        # get either 'M or 'Q' from the startdate
#        freq = 'Q' if 'Q' in startdate else 'M'
#        data.index = p.period_range(startdate, freq=freq, periods=nobs)
#
#    return data
