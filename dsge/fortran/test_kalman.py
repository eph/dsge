# import numpy as np
# import kalman, dlyap

# TT = 0.6
# RR = 1.0
# QQ = 1.0

# DD = 0.0
# ZZ = 1.0
# HH = 0.0

# Pt, info = dlyap.dlyap(TT, RR*QQ*RR)
# print 'Pt', Pt
# yy = np.random.normal(size=(1, 20))

# lik = kalman.kalman_filter(yy, TT, RR, QQ, DD, ZZ, HH, Pt)
# print 'likelihood:',  lik

# print '-----------------'
# print 'big model:'

# TT = np.loadtxt('TT.txt')
# RR = np.loadtxt('RR.txt')
# QQ = np.loadtxt('QQ.txt')
# DD = np.loadtxt('DD.txt')
# HH = np.loadtxt('HH.txt')
# ZZ = np.loadtxt('ZZ.txt')

# P0 = np.loadtxt('p0.txt')
# yy = np.loadtxt('y.txt')


# lik = kalman.kalman_filter(yy, TT, RR, QQ, DD, ZZ, HH, P0)
# print 'lik:', lik
