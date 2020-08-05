# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 04:36:32 2020

@author: yanxi
"""

import numpy as np
import matplotlib.pyplot as plt

def set_small_figure(fontsize=12):
    plt.rcParams["figure.figsize"] = [4,3]
    plt.rcParams["font.size"] = fontsize


def set_large_figure(fontsize=16):
    plt.rcParams["figure.figsize"] = [6,4.5]
    plt.rcParams["font.size"] = fontsize


# %% data generate/simulate

def calcRecoverTime(iTime, pTime, oTime, cTimeEach, ci, failProgress, sfs=False):
    ''' input time, process time, output time, checkpoint time (each one) '''
    ''' checkpoint interval, progress when fail, start from strach '''
    assert 0 < ci
    assert 0 <= failProgress <= 1
    ncp1 = int(pTime*failProgress / ci)
    ncp2 = int(pTime*(1-failProgress) / ci)
    if sfs == False:
        t1 = iTime + pTime*failProgress + ncp1*cTimeEach
        t2 = iTime + pTime - ncp1*ci + ncp2*cTimeEach + oTime
    else:
        t1 = iTime + pTime*failProgress
        t2 = iTime + pTime + oTime
    return t1 + t2
    

def prepareTimeData(x, iTime, pTime, oTime, cTimeEach, ci):
    n = len(x)
    data = np.zeros((n,3))
    data[:,0] = x
    for i in range(n):
        data[i,1] = calcRecoverTime(iTime, pTime, oTime, cTimeEach, ci, x[i], False)
        data[i,2] = calcRecoverTime(iTime, pTime, oTime, cTimeEach, ci, x[i], True)
    return data

def simulateErrorData(x, tdata, r):
    n = len(x)
    err = np.random.random((n, 2)) * r
    err *= tdata[:,1:3]
    return err
    

# %% data load
    
HEADER_TABLE=[
    'start', 'end', 
    't_none', 't_sync', 't_async', 't_vs',
    'e_none', 'e_sync', 'e_async', 'e_vs'
]

def processRecoverData(x, table, ndig=2):
    n = len(x)
    fmtStr = '%.'+str(ndig)+'f'
    xname = [fmtStr % v for v in x]
    tdata = np.zeros((n, 3))
    tdata[:,0] = x
    err = np.zeros((n,2))
    for line in table:
        s, e = line[[0,1]]
        idx = None
        if s == 0:
            idx = xname.index(fmtStr % e)
        if e == 1:
            idx = xname.index(fmtStr % s)
        if idx:
            tdata[idx, 1:] += line[[5,2]]
            err[idx, 1:] += line[[9,6]]
    return tdata, err
    
# %% draw part

def drawRecover(data, err=None, xticks=True,
                width=0.8, capsize=8, ncol=1, loc=None):
    assert data.ndim == 2
    assert data.shape[1] == 3
    ng = data.shape[0]
    nb = 2
    assert err is None or err.shape == (ng, 2)
    fp = data[:,0]
    #tfaic = data[:,1]
    #tsfs = data[:,2]
    barWidth = width/nb
    x = np.arange(ng)
    off = -width/2 + barWidth/2
    plt.figure()
    for i in range(nb):
        y = data[:,i+1]
        if err is None:
            plt.bar(x + off + barWidth*i, y, barWidth)
        else:
            plt.bar(x + off + barWidth*i, y, barWidth, yerr=err[:,i], capsize=capsize)
    if xticks == True:
        xticks = ['%.0f%%' % (v*100) for v in fp]
        plt.xticks(x, xticks)
    plt.xlabel('failure point (progress percentage)')
    plt.ylabel('running time (s)')
    plt.legend(['FAIC', 'Restart'], ncol=ncol, loc=loc)
    plt.tight_layout()

# %% main

def main(x, iTime, pTime, oTime, cTimeEach, ci):
    #tdata = prepareTimeData(x, iTime, pTime, oTime, cTimeEach, ci)
    #err = simulateErrorData(x, tdata, 0.1)
    data = np.loadtxt('../log/table-recover.txt',delimiter='\t',skiprows=1)
    tdata, err = processRecoverData(x, data, 2)
    drawRecover(tdata, err)
    