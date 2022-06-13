import numpy as np
import torch
import os
import glob
import deepdish as ddish
import random
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from nilearn.connectome import ConnectivityMeasure
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import csv
import math
from natsort import natsorted

def dynamic(feature, label, window, stride, nroi):

    dynamic_fmri = np.array([])
    dynamic_labl = np.array([])
    for i in range(len(feature)):
        subject_fmri = feature[i]
        subject_labl = label[i]
        assert subject_fmri.shape[-1] == nroi
        num_time = subject_fmri.shape[0]
        if stride == window:
            num_dynamic = math.trunc(num_time / window)
        elif stride < window:
            num_dynamic = math.trunc((num_time - window) / stride)
            if (num_dynamic * stride) + window < num_time:
                num_dynamic += 1
            # num_dynamic = math.trunc((num_time - window) / stride)
        subject_dynamic = []
        subject_dynlabl = []
        for j in range(num_dynamic):
            dynamic = subject_fmri[ stride*j : (stride*j) + window, :]
            subject_dynamic.append(dynamic)
            subject_dynlabl.append(subject_labl)

        if i == 0:
            dynamic_fmri = np.asarray(subject_dynamic, dtype='float64')
            dynamic_labl = np.asarray(subject_dynlabl, dtype='float64')
        else:
            dynamic_fmri = np.concatenate([dynamic_fmri, np.asarray(subject_dynamic, dtype='float64')], 0)
            dynamic_labl = np.concatenate([dynamic_labl, np.asarray(subject_dynlabl, dtype='float64')])

    return dynamic_fmri, dynamic_labl

def connectivity(feature, nroi):
    try:
        num_samples = feature.shape[0]
    except:
        num_samples = len(feature)

    adjacency = np.zeros((num_samples, nroi, nroi))
    Corr = ConnectivityMeasure(kind='correlation')
    for i in range(num_samples):
        adjacency[i, :, :] = Corr.fit_transform([feature[i]]) #Should be shape (time, ROI)
    return adjacency.astype('float64')

def convert_Dloader(batch_size, data, label, num_workers = 0, shuffle = True):
    data, label = torch.from_numpy(data).float(), torch.from_numpy(label).long()
    dataset = TensorDataset(data, label)
    Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return Data_loader



def convert_1D2h5(site_name, demo_path, rawd_path):

    datset_path = rawd_path + '{}_fmri.h5'.format(site_name)
    if os.path.exists(datset_path):
        dataset = ddish.io.load(datset_path)
    else:
        demo_df = pd.read_csv(demo_path)
        all_sub, all_dig = demo_df['Subject'], demo_df['Diagnosis']
        all_age, all_gen = demo_df['Age'], demo_df['Gender']

        site_sub = all_sub[all_sub.str.contains(site_name)]
        site_idx = site_sub.index
        site_dig, site_age, site_gen = all_dig[site_idx], all_age[site_idx], all_gen[site_idx]

        rawd, labl, ages, gens, sids = [], [], [], [], []
        for raw in natsorted(glob.glob('/set the paths where the .1D files exist/HO_RAW_1D/' + site_name + '/*.1D')):
            sub_id = os.path.split(raw)[-1][:-11]
            sub_idx = site_sub.index[site_sub == sub_id]
            f = open(raw)
            lines = f.readlines()[1:]
            raw_scans = []
            for j in range(len(lines)):
                list_ = []
                list_.append(np.array(list(map(float, lines[j].split()))))
                raw_scans.append(np.array(list_).squeeze(0))
            rawd.append(np.asarray(raw_scans, dtype='float64'))
            ages.append(site_age[sub_idx].item())
            gens.append(site_gen[sub_idx].item())
            sids.append(sub_id)
            labl.append(site_dig[sub_idx].item()//2)

        dataset = {'data': rawd, 'label': labl, 'gender': gens, 'age': ages, 'id': sids}
        ddish.io.save(rawd_path + '{}_fmri.h5'.format(site_name), dataset)
    return dataset

def kfold_split_ABIDE(site_name, y, sid, save_path):
    save_file = save_path + '{}_5fold_Idx.h5'.format(site_name)
    if os.path.exists(save_file):
        cv_index = ddish.io.load(save_file)
    else:
        y = np.array(y)
        sid = np.array(sid)
        if site_name in ['Caltech', 'CMU']: # small dataset
            cv_index = dict()
            skf = StratifiedKFold(n_splits=5, random_state=1210, shuffle=True)
            cv = 0
            for trainvalid_index, test_index in skf.split(sid, y):
                X_trainvalid, X_test = sid[trainvalid_index], sid[test_index]
                y_trainvalid, y_test = y[trainvalid_index], y[test_index]
                X_train, X_valid, train_index, valid_index, y_train, y_valid = train_test_split(X_trainvalid, trainvalid_index, y_trainvalid, stratify=y_trainvalid, test_size=0.1, random_state=1210)
                cv_index['f%d_tst' % (cv + 1)] = test_index
                cv_index['f%d_sid_tst' % (cv + 1)] = X_test
                cv_index['f%d_val' % (cv + 1)] = valid_index
                cv_index['f%d_sid_val' % (cv + 1)] = X_valid
                cv_index['f%d_trn' % (cv + 1)] = train_index
                cv_index['f%d_sid_trn' % (cv + 1)] = X_train
                cv += 1
        else:

            asd_idx = np.where(y == 1)[0]
            td_idx = np.where(y == 0)[0]
            random.shuffle(asd_idx), random.shuffle(td_idx)

            asd_chnk_size = int(asd_idx.shape[0] // 5)
            td_chnk_size = int(td_idx.shape[0] // 5)

            cv_index = dict()
            for cv in range(5):
                if cv != 4:
                    tst_asd = asd_idx[cv * asd_chnk_size : (cv + 1) * asd_chnk_size]
                    tst_td = td_idx[cv * td_chnk_size : (cv + 1) * td_chnk_size]
                else:
                    tst_asd = asd_idx[cv * asd_chnk_size:]
                    tst_td = td_idx[cv * td_chnk_size:]

                cv_index['f%d_tst' % (cv + 1)] = np.concatenate([tst_asd, tst_td], 0)
                cv_index['f%d_sid_tst' % (cv + 1)] = np.concatenate([sid[tst_asd], sid[tst_td]], 0)
                trnval_asd = np.setdiff1d(asd_idx, tst_asd)
                trnval_td = np.setdiff1d(td_idx, tst_td)
                val_asd = np.random.choice(trnval_asd, tst_asd.shape[0], replace=False)
                val_td = np.random.choice(trnval_td, tst_td.shape[0], replace=False)
                cv_index['f%d_val' % (cv + 1)] = np.concatenate([val_asd, val_td], 0)
                cv_index['f%d_sid_val' % (cv + 1)] = np.concatenate([sid[val_asd], sid[val_td]], 0)
                trn_asd = np.setdiff1d(trnval_asd, val_asd)
                trn_td = np.setdiff1d(trnval_td, val_td)
                cv_index['f%d_trn' % (cv + 1)] = np.concatenate([trn_asd, trn_td], 0)
                cv_index['f%d_sid_trn' % (cv + 1)] = np.concatenate([sid[trn_asd], sid[trn_td]], 0)

        ddish.io.save(save_file, cv_index)

    assert np.intersect1d(cv_index['f1_sid_trn'], cv_index['f1_sid_tst']).shape[0] == 0
    assert np.intersect1d(cv_index['f1_sid_trn'], cv_index['f1_sid_val']).shape[0] == 0
    assert np.intersect1d(cv_index['f1_sid_val'], cv_index['f1_sid_tst']).shape[0] == 0

    return cv_index



def dataloader(args, fold):

    # Directory paths
    main_path = 'Data Directory paths'
    demo_file = main_path + 'ABIDEI_qc_filtered-lists.csv'
    rawd_path = main_path + 'HO110/RAW/'
    cond_path = main_path + 'HO110/CON/'
    idxd_path = main_path + 'HO110/IDX/'
    sites_to_load = ['NYU', 'UCLA_1', 'UCLA_2', 'UM_1', 'UM_2', 'USM',
                     'Leuven_1', 'Leuven_2', 'Pitt', 'Yale', 'Caltech', 'CMU',
                     'KKI', 'MaxMun', 'OHSU', 'Olin', 'SBL', 'SDSU', 'Stanford', 'Trinity']

    trnc_a, valc_a, tstc_a = np.empty((0, args.input_size, args.input_size)), \
                             np.empty((0, args.input_size, args.input_size)), \
                             np.empty((0, args.input_size, args.input_size))
    trnl_a, vall_a, tstl_a = np.array([]), np.array([]), np.array([])

    for site in sites_to_load:
        cons_file = cond_path + 'Static/{}_conn.h5'.format(site)
        rawd_dict = convert_1D2h5(site, demo_file, rawd_path)

        if not os.path.exists(cons_file):
            allc = connectivity(rawd_dict['data'], args.input_size)
            conn_dict = {'data': allc, 'label': rawd_dict['label'], 'gender': rawd_dict['gender'], 'age': rawd_dict['age'], 'id': rawd_dict['id']}
            ddish.io.save(cons_file, conn_dict)
        else:
            conn_dict = ddish.io.load(cons_file)

        fold_dict = kfold_split_ABIDE(site, rawd_dict['label'], rawd_dict['id'], idxd_path)
        tr_sid, vl_sid, te_sid = fold_dict['f%d_sid_trn' % fold], fold_dict['f%d_sid_val' % fold], fold_dict['f%d_sid_tst' % fold]

        # To check trnr == trnr2
        # tr_id, vl_id, te_id = fold_dict['f%d_trn' % fold], fold_dict['f%d_val' % fold], fold_dict['f%d_tst' % fold]
        # trnr = [rawd_dict['data'][j] for j in tr_id]
        # valr = [rawd_dict['data'][j] for j in vl_id]
        # tstr = [rawd_dict['data'][j] for j in te_id]

        trni = [rawd_dict['id'].index(k) for k in tr_sid]
        trnl = [rawd_dict['label'][j] for j in trni]
        trnr = [rawd_dict['data'][j] for j in trni]

        trnc = np.empty((0, args.input_size, args.input_size))
        trnll = np.array([])
        window_stride = [[30, 15], [50, 25], [70, 35], [100, 50]] # window, stride
        for ws in window_stride:
            trnrr, b = dynamic(trnr, trnl, ws[0], ws[1], args.input_size)
            trnc_tmp = connectivity(trnrr, args.input_size)
            trnc = np.vstack((trnc, trnc_tmp))
            trnll = np.concatenate((trnll, b))

        trnc_s = np.array([conn_dict['data'][j] for j in trni], dtype='float64')
        trnl_s = np.array([conn_dict['label'][j] for j in trni])

        sw_tr_idx_rand = np.random.choice(trnc.shape[0], int(trnc.shape[0]/2))
        trnc_d = trnc[sw_tr_idx_rand, ...]
        trnl_d = trnll[sw_tr_idx_rand, ...]

        # vali = [np.where(conn_dict['id'] == k)[0][0] for k in vl_sid]
        # tsti = [np.where(conn_dict['id'] == k)[0][0] for k in te_sid]
        vali = [conn_dict['id'].index(k) for k in vl_sid]
        tsti = [conn_dict['id'].index(k) for k in te_sid]
        valc = np.array([conn_dict['data'][j] for j in vali], dtype='float64')
        tstc = np.array([conn_dict['data'][j] for j in tsti], dtype='float64')
        vall = np.array([conn_dict['label'][j] for j in vali])
        tstl = np.array([conn_dict['label'][j] for j in tsti])

        trnc_a = np.vstack((trnc_a, trnc_d, trnc_s))
        valc_a = np.vstack((valc_a, valc))
        tstc_a = np.vstack((tstc_a, tstc))
        trnl_a = np.concatenate([trnl_a, trnl_d, trnl_s])
        vall_a = np.concatenate([vall_a, vall])
        tstl_a = np.concatenate([tstl_a, tstl])

    train_loader = convert_Dloader(args.bs, trnc_a, trnl_a, num_workers=0, shuffle=True)
    val_loader = convert_Dloader(args.bs, valc_a, vall_a, num_workers=0)
    test_loader = convert_Dloader(args.bs, tstc_a, tstl_a, num_workers=0)
    # If high computing power,
    # val_loader = convert_Dloader(valc_a.shape[0], valc_a, vall_a, num_workers=0)
    # test_loader = convert_Dloader(tstc_a.shape[0], tstc_a, tstl_a, num_workers=0)
    return train_loader, val_loader, test_loader

