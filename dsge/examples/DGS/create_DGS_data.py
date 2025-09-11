#!/usr/bin/env python
# coding: utf-8
import numpy as np
import datetime 
import pandas_datareader as web

start = datetime.datetime(1965, 1, 1)
end = datetime.datetime(2012, 12, 31)


# fd
# 1. [[ffdsf]] 

# In[3]:


# Load FRED keys
# 
fred = dict()
fred["gdp"]   = "GDPC1"         # The base year has changed to 2009
fred["inv"]   = "FPI"
fred["cons"]  = "PCEC"
fred["inf"]   = "GDPDEF"
fred["pop"]   = "CNP16OV"       # This is BLS LNS10000000
fred["emp"]   = "CE16OV"                
fred["hours"] = "PRS85006023"
fred["wages"] = "COMPNFB"       # This is BLS PRS85006103
fred["ffr"]   = "FEDFUNDS"
fred['baa'] = 'BAA'
fred['rg10f'] = 'GS10'


data = web.DataReader(fred.values(), "fred", start, end)

data = data.resample('Q').mean().to_period('Q')

# # Get SPF data -- can't access from internet easily at board
# spf_piexp_file = 'Mean_PCE10_Level_SPF.xls'
# spf_piexp = p.ExcelFile(spf_piexp_file).parse('Mean_Level', header=0, na_values='-999')
# to_period  = lambda (row): str(int(row['year'])) + '-' + str(int(row['quarter']*3))
# per = spf_piexp.apply(to_period, axis=1)
# spf_piexp.index = p.PeriodIndex(per, freq='Q')
# data['SPF_IND'] = spf_piexp['PCE10']

data['LNSindex'] = data['CNP16OV'] / data['CNP16OV']['1992Q3']
data['EMPindex'] = data['CE16OV'] / data['CE16OV']['1992Q3'] * 100




# Conversions, et cetera
data['consumption'] = np.log((data['PCEC']/data['GDPDEF']) / data['LNSindex']) * 100
data['investment']  = np.log((data['FPI']/data['GDPDEF']) / data['LNSindex']) * 100
data['output']      = np.log(data['GDPC1'] / data['LNSindex']) * 100
data['hours']       = np.log(((data['PRS85006023'] * data['EMPindex'])/100)/data['LNSindex']) * 100
data['inflation']   = np.log(data['GDPDEF'] / data['GDPDEF'].shift(1)) * 100
data['real wage']   = np.log(data['COMPNFB'] / data['GDPDEF']) * 100
data['interest rate'] = data['FEDFUNDS'] / 4



# now, growth rates, in our mnenomics
obs = ['ygr', 'cgr', 'igr', 'wgr', 'lnh', 'pinfobs', 'robs', 'ptrobs', 'spreadobs_baa']

data['ygr'] = data['output'].diff()
data['cgr'] = data['consumption'].diff()
data['igr'] = data['investment'].diff()
data['wgr'] = data['real wage'].diff()
data['lnh'] = data['hours'] - data['hours'].mean()
data['pinfobs'] = data['inflation']
data['robs'] = data['interest rate']
data['ptrobs'] = np.nan #data['expected infl']
data['spreadobs_baa'] = (data['BAA'] - data['GS10'])/4
data['1965Q2':][obs].to_csv('yy.txt', header=False, index=False, sep=',')



# In[8]:

