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

def prune_heading(dr, dc, th=0.005):
    n = dr.shape[0]
    if th is None:
        th = np.inf
    p = 0
    while p < n-1 and (dr[p, 1] < dr[p+1, 1] or abs(dr[p, 2] / dr[p, 1]) > th):
        p += 1
    return dr[p:, :], dc[p:, :]

def parse_name(fn):
    reg = re.compile('''(.+)-(\d+)-(\d+)-(none|sync|async|vs)''')
    m = reg.match(fn)
    if m is None:
        return m
    else:
        return {'name':m.group(0), 'ds':m.group(1), 'nw':int(m.group(2)),
                'ci':int(m.group(3)), 'ctype':m.group(4)}

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

def get_ds(fld, nw=None, ci=None, ctype=None):
    l = os.listdir(fld)
    ds_list = []
    for fn in l:
        m = parse_name(fn)
        if nw is not None and m['nw'] != nw:
            continue
        if ci is not None and m['ci'] != ci:
            continue
        if ctype is not None and m['ctype'] != ctype:
            continue
        if m['ds'] not in ds_list:
            ds_list.append(m['ds'])
    return ds_list    

def load_files(fld, ds, nw=None, ci=None, ctype=None, th=None):
    dr_list = []
    dc_list = []
    fn_list, m_list = get_file_list(fld, ds, nw, ci, ctype);
    for fn in fn_list:
        dr = read_rmse_file(fld+'/'+fn+'-rmse.csv')
        dc = read_cp_file(fld+'/'+fn+'-cp.csv')
        if th:
            dr, dc = prune_heading(dr, dc, th)
        dr_list.append(dr)
        dc_list.append(dc)
    return fn_list, m_list, dr_list, dc_list


# %% data prepare

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

HEAD_GROUP=['ds', 'nw', 'ci', 't_none',
            't_sync', 't_async', 't_vs',
            'nc_sync', 'nc_async', 'nc_vs',
            'tc_sync', 'tc_async', 'tc_vs']

def group_info(fn_list, info_list):
    n = len(fn_list)
    res = {}
    dsmap = {}
    dsmap_r = {}
    for i in range(n):
        fn = fn_list[i]
        info = info_list[i]
        m = parse_name(fn)
        h = '%s-%d-%d' % (m['ds'], m['nw'], m['ci'])
        if h in res:
            v = res[h]
        else:
            if m['ds'] in dsmap:
                ds = dsmap[m['ds']]
            else:
                ds = len(dsmap)
                dsmap[m['ds']] = ds
                dsmap_r[ds] = m['ds']
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
    r = r.tolist()
    for i in range(len(r)):
        r[i][0] = dsmap_r[int(r[i][0])]
    return r

def dump_group_table(group_list, out_file, append=False):
    mode = 'a' if append else 'w'
    with open(out_file, mode) as f:
        for g in group_list:
            s = '%s\t%d\t%d' + '\t%f'*4 + '\t%d'*3 + '\t%f'*3 + '\n'
            f.write(s % tuple(g))

# %% main

def main(ofile. omode, ifld, ds. nw=None, ci=None, th=None):
    print('run for data set:', ds)
    fn_list,m_list,dr_list,dc_list=load_files(ifld, ds,nw, ci, th)
    rth=get_max_ending_rmse(dr_list)
    print('rmse threshold: ',rth)
    info_list=get_info_all(dr_list, dc_list, rth)
    group=group_info(fn_list, info_list)
    print('number of lines:', len(group))
    dump_group_table(group, ofile)


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc <= 2:
        print('Make table from csv files.')
        print('usage: <in-folder> <out-file> [th] [ds] [nw] [ci]')
        print('  [th]: (=0.005) threashold for pruning the head of data')
        print('  [ds]: (=None) data set name, filter option')
        print('  [nw]: (=None) number of workers, filter option')
        print('  [ci]: (=None) checkpoint interval, filter option')
    else:
        ifld = sys.argv[1]
        ofile = sys.argv[2]
        th = float(sys.argv[3]) if argc > 3 else 0.005
        ds = sys.argv[4] if argc > 4 else None
        nw = int(sys.argv[5]) if argc > 5 else None
        ci = int(sys.argv[6]) if argc > 6 else None
        if ds is None:
            ds_list = get_ds(ifld, nw, ci)
            omode = 'w'
            for ds in ds_list:
                main(ofile, omode, ifld, ds, nw, ci, th)
                omode = 'a'
        else:
            main(ofile, 'w', ifld, ds, nw, ci, th)
        print('finish')
        
        