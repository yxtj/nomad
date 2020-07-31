# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 09:05:53 2020

@author: yanxi
"""

import os,sys
import pandas
import re
import numpy as np

# %% data loading and preprocessing functions
        
HEADER_RMSE=['time', 'rmse', 'diff', 'update']
HEADER_CP=['n', 'time', 'cost']

def read_rmse_file(fn):
    df = pandas.read_csv(fn, header=None)
    return df.to_numpy()

def read_cp_file(fn):
    with open(fn) as f:
        if f.seek(0,2) == 0:
            return np.array([[],[]]).reshape(0,2)
        else:
            f.seek(0,0)
            df = pandas.read_csv(f, header=None, usecols=[1,2])
            return df.to_numpy()

def construct_file_names(fld, ds, nw, ci, ctype):
    fpre = '%s/%s-%d-%d-%s' % (fld, ds, nw, ci, ctype)
    frmse = fpre + '-rmse.csv'
    fcp = fpre + '-cp.csv'
    return (frmse, fcp)

def load_pair(fld, ds, nw, ci, ctype):
    fr, fc = construct_file_names(fld, ds, nw, ci, ctype)
    dr = read_rmse_file(fr)
    dc = read_cp_file(fc)
    return dr, dc

def prune_heading(dr, th=0.005):
    n = dr.shape[0]
    if th is None:
        th = np.inf
    flag1 = dr[:,2] > 0
    flag2 = dr[:,2]/dr[:,1] > th
    flag = np.logical_or(flag1, flag2)
    p = flag.nonzero()[0]
    if p.size == 0:
        return dr
    return dr[p[-1]:]

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

def get_ds(fld, nw=None, ci=None, ctype=None):
    l = os.listdir(fld)
    ds_list = []
    for fn in l:
        m = parse_name(fn)
        if m is None:
            continue
        if nw is not None and m['nw'] != nw:
            continue
        if ci is not None and m['ci'] != ci:
            continue
        if ctype is not None and m['ctype'] != ctype:
            continue
        if m['ds'] not in ds_list:
            ds_list.append(m['ds'])
    ds_list.sort(key=lambda n:ds2int(n))
    return ds_list

def get_file_list(fld, ds, nw=None, ci=None, ctype=None):
    l = os.listdir(fld)
    fn_list = []
    m_list = []
    for fn in l:
        m = parse_name(fn)
        if m is None:
            continue
        n = m['name']
        if n in fn_list or m['ds'] != ds:
            continue
        if nw is not None and m['nw'] != nw:
            continue
        if ci is not None and m['ci'] != ci:
            continue
        if ctype is not None and m['ctype'] != ctype:
            continue
        fn_list.append(n)
        m_list.append(m)
    return fn_list, m_list

def load_files(fld, ds, nw=None, ci=None, ctype=None, th=None):
    dr_list = []
    dc_list = []
    fn_list, m_list = get_file_list(fld, ds, nw, ci, ctype);
    for fn in fn_list:
        dr = read_rmse_file(fld+'/'+fn+'-rmse.csv')
        dc = read_cp_file(fld+'/'+fn+'-cp.csv')
        if th:
            dr = prune_heading(dr, th)
        dr_list.append(dr)
        dc_list.append(dc)
    return fn_list, m_list, dr_list, dc_list


# %% data prepare (local tm)
    
HEADER_GROUP=['ds', 'nw', 'ci', 't_none',
              't_sync', 't_async', 't_vs',
              'nc_sync', 'nc_async', 'nc_vs',
              'tc_sync', 'tc_async', 'tc_vs']

class Item():
    def __init__(self, gn, m):
        self.gn = gn
        self.size = ds2int(m['ds'])
        self.nw = int(m['nw'])
        self.ci = int(m['ci'])
        self.dr_none = None
        self.dr_sync = None
        self.dc_sync = None
        self.dr_async = None
        self.dc_async = None
        self.dr_vs = None
        self.dc_vs = None
    
    def __repr__(self):
        return self.gn
    
    def set_none(self, dr):
        self.dr_none = dr
        
    def set_sync(self, dr, dc):
        self.dr_sync = dr
        self.dc_sync = dc
        
    def set_async(self, dr, dc):
        self.dr_async = dr
        self.dc_async = dc
    
    def set_vs(self, dr, dc):
        self.dr_vs = dr
        self.dc_vs = dc
    
    def __get_ending_rmse__(self):
        r = 0
        for dr in [self.dr_none, self.dr_sync, self.dr_async, self.dr_vs]:
            if dr is not None and dr[-1,1] > r:
                r = dr[-1,1]
        return r
    
    def __get_time_at_rmse__(self, dr, dc, rth):
        if dr is None:
            return 0, 0, 0, 0
        flag = dr[:,1] > rth
        p = np.argmin(flag)
        # flag[p] == False, flag[p-1] == True
        found = p < dr.shape[0]
        if not found:
            return None
        t = dr[p,0]
        if dc is not None and dc.size > 0:
            flag2 = dc[:,0] < t
            ncp = sum(flag2)
            tcost = sum(dc[flag2,1])
        else:
            ncp = 0
            tcost = 0.0
        return t, dr[p, 1], ncp, tcost
    
    def get_info(self, rth=None):
        if rth is None:
            rth = self.__get_ending_rmse__()
            print('  rmse threshold of %s is %f' % (self.gn, rth))
        v = self.__get_time_at_rmse__(self.dr_none, None, rth)
        info = np.zeros(4+3*3)
        info[:4] = [self.size, self.nw, self.ci, v[0]]
        idx = np.array([4,7,10])
        dr_list = [self.dr_sync, self.dr_async, self.dr_vs]
        dc_list = [self.dc_sync, self.dc_async, self.dc_vs]
        for i in range(3):
            v = self.__get_time_at_rmse__(dr_list[i], dc_list[i], rth)
            info[idx + i] = [v[0], v[2], v[3]]
        return info
    
    
def group_lists(fn_list, m_list, dr_list, dc_list):
    n = len(fn_list)
    res = {}
    for i in range(n):
        m = m_list[i]
        gn = '%s-%d-%d' % (m['ds'], m['nw'], m['ci'])
        if gn in res:
            v = res[gn]
        else:
            v = Item(gn, m)
        if m['ctype'] == 'none':
            v.set_none(dr_list[i])
        elif m['ctype'] == 'sync':
            v.set_sync(dr_list[i], dc_list[i])
        elif m['ctype'] == 'async':
            v.set_async(dr_list[i], dc_list[i])
        elif m['ctype'] == 'vs':
            v.set_vs(dr_list[i], dc_list[i])
        res[gn] = v
    # sort
    r = np.arange(len(res))
    r = list(res.values())
    r = sorted(r, key=lambda d:d.ci)
    r = sorted(r, key=lambda d:d.nw)
    r = sorted(r, key=lambda d:d.size)
    return r

def item2table(item_list):
    res = np.vstack([item.get_info() for item in item_list])
    return res
    
# %% data prepare (global tm)
    
def cut_list_by_dataset(m_list, data_list):
    assert len(m_list) == len(data_list)
    res = []
    ds = None
    tmp = []
    n = len(m_list)
    for i in range(n):
        m = m_list[i]
        d = data_list[i]
        if ds is None or ds == m['ds']:
            tmp.append(d)
        else:
            res.append(tmp)
            tmp = [d]
        ds = m['ds']
    if len(tmp) != 0:
        res.append(tmp)
    return res
    

def get_max_ending_rmse(dr_list):
    r = 0
    for dr in dr_list:
        if dr[-1,1] > r:
            r = dr[-1,1]
    return r

def get_info_at_rmse(dr, dc, rth):
    flag = dr[:,1] > rth
    p = np.argmin(flag)
    # flag[p] = False, flag[p-1] = True
    found = p < dr.shape[0]
    if not found:
        return None
    t = dr[p,0]
    if dc.size > 0:
        flag2 = dc[:,0] <= t
        ncp = sum(flag2)
        tcost = sum(dc[flag2,1])
    else:
        ncp = 0
        tcost = 0.0
    return t, dr[p, 1], ncp, tcost
    
def get_ending_time_at(dr_list, rth):
    res = []
    for dr in dr_list:
        p = np.argmin(dr[:,1] > rth)
        res.append(dr[p,0])
    return res
    
def get_info_all(dr_list, dc_list, rth):
    res = []
    for dr, dc in zip(dr_list, dc_list):
        tmp = get_info_at_rmse(dr, dc, rth)
        res.append(tmp)
    return np.array(res)

HEADER_GROUP=['ds', 'nw', 'ci', 't_none',
              't_sync', 't_async', 't_vs',
              'nc_sync', 'nc_async', 'nc_vs',
              'tc_sync', 'tc_async', 'tc_vs']

def group_info_to_table(fn_list, info_list):
    n = len(fn_list)
    res = {}
    for i in range(n):
        fn = fn_list[i]
        info = info_list[i]
        m = parse_name(fn)
        h = '%s-%d-%d' % (m['ds'], m['nw'], m['ci'])
        if h in res:
            v = res[h]
        else:
            ds = ds2int(m['ds'])
            v = [ds, m['nw'], m['ci'], 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0]
        if m['ctype'] == 'none':
            v[3] = info[0]
        elif m['ctype'] == 'sync':
            v[4] = info[0]
            v[7] = info[2]
            v[10] = info[3]
        elif m['ctype'] == 'async':
            v[5] = info[0]
            v[8] = info[2]
            v[11] = info[3]
        elif m['ctype'] == 'vs':
            v[6] = info[0]
            v[9] = info[2]
            v[12] = info[3]
        res[h] = v
    # sort
    r = np.array(list(res.values()))
    for i in range(2, -1, -1):
        r = r[r[:,i].argsort(kind='stable')]
    return r


# %% main
    
def format_one_row(line):
    s = '%d\t%d\t%d' + '\t%.2f'*4 + '\t%d'*3 + '\t%f'*3 + '\n'
    return s % tuple(line)

def dump_group_table(group_list, out_file, append=False):
    mode = 'a' if append else 'w'
    with open(out_file, mode) as f:
        if append == False:
            f.write('\t'.join(HEADER_GROUP))
            f.write('\n')
        for g in group_list:
            line = format_one_row(g)
            f.write(line)
    
def main(ofile, append, method, ifld, ds, nw=None, ci=None, th=None):
    print('run for data set:', ds)
    fn_list,m_list,dr_list,dc_list=load_files(ifld, ds, nw, ci, None, th)
    if method == 'group':
        item_list=group_lists(fn_list,m_list,dr_list,dc_list)
        group=item2table(item_list)
    else:
        if method == 'global':
            rth=get_max_ending_rmse(dr_list)
        else:
            rth=float(method)
        print('rmse threshold: ',rth)
        #get_ending_time_at(dr_list,rth)
        info_list=get_info_all(dr_list, dc_list, rth)
        group=group_info_to_table(fn_list, info_list)
    print('number of lines:', len(group))
    dump_group_table(group, ofile, append)
    
# %% running

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc <= 2:
        print('Make table from csv files.')
        print('usage: <in-folder> <out-file> [rmse-mthd] [th] [ds] [nw] [ci]')
        print('  [rmse-mthd]: (=global) method to find the terminination time.' +
              ' Support: "global", "group", a float number.' +
              ' For global/group automatically finding, given RMSE value.')
        print('  [th]: (=0.005) threashold for pruning the head of data')
        print('  [ds]: (=None) data set name, filter option')
        print('  [nw]: (=None) number of workers, filter option')
        print('  [ci]: (=None) checkpoint interval, filter option')
    else:
        ifld = sys.argv[1]
        ofile = sys.argv[2]
        m = re.match('''^(global|group|\d+(?:\.\d*)?)$''',sys.argv[3], re.IGNORECASE)
        method = m[0] if argc > 3 else 'global'
        th = float(sys.argv[4]) if argc > 4 else 0.005
        ds = sys.argv[5] if argc > 5 else None
        nw = int(sys.argv[6]) if argc > 6 else None
        ci = int(sys.argv[7]) if argc > 7 else None
        if ds is None:
            ds_list = get_ds(ifld, nw, ci)
            append = False
            for ds in ds_list:
                main(ofile, append, method, ifld, ds, nw, ci, th)
                append = True
        else:
            main(ofile, False, method, ifld, ds, nw, ci, th)
        print('finish')
        
        