# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:43:13 2020

@author: yanxi
"""

# %% prepare

import re
import numpy as np
import matplotlib.pyplot as plt


def set_small_figure():
    plt.rcParams["figure.figsize"] = [4,3]
    plt.rcParams["font.size"]=12


def set_large_figure():
    plt.rcParams["figure.figsize"] = [6,4.5]
    plt.rcParams["font.size"] = 16


HEADER_GROUP=['ds', 'nw', 'ci', 't_none',
              't_sync', 't_async', 't_vs',
              'nc_sync', 'nc_async', 'nc_vs',
              'tc_sync', 'tc_async', 'tc_vs']

def parse_name(fn):
    reg = re.compile('''(.+)-(\d+)-(\d+)-(none|sync|async|vs)''')
    m = reg.match(fn)
    if m is None:
        return m
    else:
        return {'name':m.group(0), 'ds':m.group(1), 'nw':int(m.group(2)),
                'ci':int(m.group(3)), 'ctype':m.group(4)}

def ds2int(ds):
    f = 1
    if ds[-1] in ['k', 'K']:
        f = 1000
        ds = ds[:-1]
    elif ds[-1] in ['m', 'M']:
        f = 1000*1000
        ds = ds[:-1]
    r = int(ds)
    return r * f

# %% data selection
    
def select(data, ds=None, nw=None, ci=None):
    assert ds is not None or nw is not None or ci is not None
    if ds is not None:
        if isinstance(ds, str):
            ds = ds2int(ds)
        f = data[:,0] == ds
        data = data[f,:]
    if nw is not None:
        f = data[:,1] == nw
        data = data[f,:]
    if ci is not None:
        f = data[:,2] == ci
        data = data[f,:]
    return data

# %% draw functions

def drawRunTime(data, width=0.8, xlbl=None, xticks=None, ncol=1, loc=None):
    dp = data[:, [3,4,5,6]]
    #plt.plot(dp)
    ng = dp.shape[0]
    nb = 4
    barWidth = width/nb
    x = np.arange(ng)
    off = -width/2 + barWidth/2
    plt.figure()
    for i in range(nb):
        y = dp[:,i]
        plt.bar(x + off + barWidth*i, y, barWidth)
    if xlbl is None and xticks is None:
        plt.xticks(x, xticks)
        plt.xlabel(xlbl)
    plt.ylabel('running time (s)')
    #plt.legend(['None','Sync','Async','VS'], ncol=ncol, loc=loc)
    plt.tight_layout()
    

def drawCmpOne(line, average=True, relative=False,
               width=0.6, atxlbl=True, ncol=1, loc=None):
    dnone = line[3]
    dcp = line[4:7] - dnone
    ncp = line[7:10]
    if average:
        dcp = dcp / ncp
    if relative:
        dcp = dcp / dnone * 100
    x = 0
    plt.figure()
    for i in range(3):
        x = i
        y = dcp[i]
        plt.bar(x, y, width)
    if average:
        if relative:
            ylbl = 'average overhead ratio (%)'
        else:
            ylbl = 'overhead per checkpoint (s)'
    else:
        if relative:
            ylbl = 'ratio of overhead (%)'
        else:
            ylbl = 'overhead time (s)'
    plt.ylabel(ylbl)
    if atxlbl:
        plt.xticks(np.arange(3), ['Sync','Async','VS'])
    else:
        plt.legend(['Sync','Async','FAIC'], ncol=ncol, loc=loc)
    plt.tight_layout()


def drawOverhead(data, average=True, relative=False, xlbl=None, xticks=None, 
                 width=0.8, capsize=8, ncol=1, loc=None):
    dnone = data[:, 3].reshape(-1, 1)
    idx = np.array([4,6,5])
    dp = data[:, idx] - dnone
    ncp = data[:,idx+3]
    #plt.plot(dp)
    if average:
        dp = dp / ncp
    if relative:
        dp = dp / dnone * 100
    ng = dp.shape[0]
    nb = 3
    barWidth = width/nb
    x = np.arange(ng)
    off = -width/2 + barWidth/2
    plt.figure()
    for i in range(nb):
        y = dp[:,i]
        plt.bar(x + off + barWidth*i, y, barWidth)
    if xticks is None:
        plt.xticks(x, xticks)
    if xlbl is None:
        xticks = ['Sync','VS','Async']
    plt.xlabel(xlbl)
    if average:
        if relative:
            ylbl = 'average overhead ratio (%)'
        else:
            ylbl = 'overhead per checkpoint (s)'
    else:
        if relative:
            ylbl = 'ratio of overhead (%)'
        else:
            ylbl = 'overhead time (s)'
    plt.ylabel(ylbl)
    #plt.legend(['Sync','FAIC','Async'], ncol=ncol, loc=loc)
    plt.tight_layout()
    
def drawOverheadGroup(data, relative=False, xlbl=None, xticks=None, 
                      width=0.6, capsize=8):
    assert data.shape[0] > 1
    dnone = data[:, 3].reshape(-1, 1)
    idx = np.array([4,6,5])
    dp = data[:, idx] - dnone
    ncp = data[:,idx+3]
    #plt.plot(dp)
    dp = dp / ncp
    if relative:
        dp = dp / dnone * 100
    y = dp.mean(0)
    err = dp.std(0)
    nb = 3
    x = np.arange(nb)
    plt.figure()
    for i in range(nb):
        plt.bar(x[i], y[i], yerr=err[i], width=width, capsize=capsize)
    if xticks is None:
        xticks = ['Sync','VS','Async']
    plt.xticks(x, xticks)
    if xlbl is None:
        xlbl = 'checkpoint method'
        plt.xlabel(xlbl)
    if relative:
        ylbl = 'average overhead ratio (%)'
    else:
        ylbl = 'overhead per checkpoint (s)'
    plt.ylabel(ylbl)
    #plt.legend(['Sync','FAIC','Async'], ncol=ncol, loc=loc)
    plt.tight_layout()

def drawScaleWorker(data, overhead=True, average=True,
                    logx=False, logy=False, ncol=1, loc=None):
    x = data[:,1]
    idx = np.array([4,6,5])
    if overhead:
        y0 = data[:,3].reshape(data.shape[0], 1)
        y = data[:,idx] - y0
        if average:
            y /= data[:,idx+3]
    else:
        y = data[:,[3,4,5,6]]
    plt.figure()
    plt.plot(x, y)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.xlabel('number of worker')
    if overhead:
        ylbl = 'average overhead (s)' if average else 'overhead time (s)'
        lgd = ['Sync','FAIC','Async']
    else:
        ylbl = 'running time (s)'
        lgd = ['None', 'Sync','FAIC','Async']
    plt.ylabel(ylbl)
    plt.legend(lgd, ncol=ncol, loc=loc)
    plt.tight_layout()

# %% main

def main():
    data=np.loadtxt('../log/table-2.txt',delimiter='\t',skiprows=1)
    data1=select(data, '10k')
    