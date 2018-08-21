from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from datetime import datetime as dt
import re
import copy
import os


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def init_weights(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    return W.astype(np.float32)


def init_bias(M):
    b = np.zeros(M)
    return b.astype(np.float32)


def add_label(df):  # clip values and normalize/standardize for regression
    df['maxreturn'] = (df.exit_price - df.entry_price) / df.zatr
    # df['label'] = (df.maxreturn > 0.05).astype(int)
    t0 = df.maxreturn.quantile(0.33)
    t1 = df.maxreturn.quantile(0.67)
    df['label'] = (df.maxreturn > t0).astype(int) + (df.maxreturn > t1).astype(int)
    print(df.maxreturn.describe())
    print('Percentage of Target Trades', len(df[df.label == 2]) / len(df))

    return df


def readFile(filename, sample, start_date=None):
    saved = os.path.isfile(filename)
    if saved and '.feather' in filename:
        df = pd.read_feather(filename)
    else:
        df = pd.read_csv(filename)

    if start_date:
        df = df[pd.to_datetime(df.entry_time) > pd.to_datetime(start_date)]

    cols = list()
    for i in df.columns:
        if i[:14] == 'entry_collect.':
            cols.append(i[14:])
        else:
            cols.append(i)
    df.columns = cols

    if sample and sample < len(df):
        df = df.sample(sample)
    df = df.sort_values('entry_time')

    return df


def save_arrays(data, filename):
    for k in data.keys():
        file_path = '.'.join(filename.split('.')[:-1]) + '_' + k + '.npy'
        np.save(file_path, data[k])
        print('Saved', k)


def load_arrays(filename, norm=False):
    data = dict()
    splits = filename.split('/')
    dirname = '/'.join(splits[:-1]) + '/'
    f0 = splits[-1].split('.')[-2]
    for i in os.listdir(dirname):
        if not norm:
            if '.npy' in i and f0 in i and 'norm' not in i and 'features' not in i and 'idxs' not in i:
                idx = len(f0) + 1
                fname = i[idx:-4]
                data[fname] = np.load(dirname + i)
                print('Loaded', fname, data[fname].shape)
        else:
            if '.npy' in i and f0 in i and 'norm' in i and 'features' not in i and 'idxs' not in i:
                idx = len(f0) + 1
                fname = i[idx:-4]
                data[fname] = np.load(dirname + i)
                print('Loaded', fname, data[fname].shape)

    return data


def drop_feats(df, feat_dict, null_drop=0.1):
    features = list()
    for k in feat_dict.keys():
        features.extend(feat_dict[k]['feats'])
    features = np.unique(features)

    drop = list()
    for i in df.index:
        p = df.loc[i, features].isnull().sum() / float(len(features))
        if p > null_drop:
            drop.append(i)
    print('Dropping', len(drop), 'samples')
    df = df.drop(drop)

    df = df.replace(np.inf, 10e30)
    df = df.replace(-np.inf, -10e30)

    return df


def separate_balance_data(df, split_data, balance=True, downsample=False, save_test=False):
    if type(split_data) == str:
        s = pd.to_datetime(split_data)
        df['entry_time'] = pd.to_datetime(df.entry_time)
        df, df_test = df[df.entry_time < s], df[df.entry_time >= s]
    else:
        s = int(len(df) * split_data)
        df, df_test = df[:s], df[s:]

    if balance:
        sizes = list()
        class_dict = dict()
        classes = np.unique(df.label)
        for c in classes:
            size = (df.label == c).sum()
            class_dict[str(c)] = size
            sizes.append(size)
        max_size = max(sizes)
        min_size = min(sizes)
        p_diff = abs(np.log(max_size / min_size))
        if p_diff > 0.05:
            if not downsample:
                for k, v in class_dict.items():
                    up_sample = max_size - v
                    if up_sample == 0:
                        continue
                    d = pd.DataFrame()
                    c = df[df.label == int(k)]
                    multi = int(up_sample // len(c))
                    remain = int(up_sample % len(c))
                    for _ in range(multi):
                        d = d.append(c)
                    if remain:
                        c = c.sample(remain)
                        d = d.append(c)
                    df = df.append(d)
            else:
                d = pd.DataFrame()
                for k, v in class_dict.items():
                    c = df[df.label == int(k)]
                    if v > min_size:
                        c = c.sample(min_size)
                    d = d.append(c)
                df = d
            df = df.sort_values('entry_time')
            if 'level_0' in df.columns:
                df.drop('level_0', axis=1, inplace=True)
            df = df.reset_index()

    y = np.array(df['label']).astype(np.int64)
    y_test = np.array(df_test['label']).astype(np.int64)
    n_check0 = len(np.argwhere(np.isnan(y)))
    n_check1 = len(np.argwhere(np.isnan(y_test)))
    assert n_check0 == 0
    assert n_check1 == 0

    if save_test:
        filename = '.'.join(save_test.split('.')[:-1]) + '_testidxs.csv'
        df_test[['symbol', 'entry_time', 'label', 'mtm_pl', 'zl1t']].to_csv(filename)

    return df, df_test, y, y_test


def get_features(df, save_feats=True):
    data = dict()
    data['time_0'] = {'scale': 'normalize', 'time_steps': 14, 'clip_inputs': True, 'reverse': False}
    data['time_1'] = {'scale': 'normalize', 'time_steps': 20, 'clip_inputs': False, 'reverse': True}
    data['time_2'] = {'scale': 'normalize', 'time_steps': 20, 'clip_inputs': False, 'reverse': True}
    data['time_3'] = {'scale': 'normalize', 'time_steps': 20, 'clip_inputs': False, 'reverse': True}
    data['static_mins'] = {'scale': 'normalize', 'time_steps': 0, 'clip_inputs': True, 'reverse': False}
    data['time_0']['feats'] = list()
    data['time_1']['feats'] = list()
    data['time_2']['feats'] = list()
    data['time_3']['feats'] = list()
    data['static_mins']['feats'] = list()

    static_feats = list(np.load('D:/data_files/staticfeatsimballong.npy'))
    for i in df.columns:
        if i[:4] == 'nyi_':
            data['time_0']['feats'].append(i)
        elif 'tsd_mama' in i or 'tsd_fama' in i or 'tsd_sma' in i or 'tsd_upperbb' in i or 'tsd_macdhist' in i:
            continue
        elif i[:4] == 'hlow' or i[:5] in ['htsd_', 'hhigh', 'hvwap', 'mhlow'] or \
                i[:6] in ['hclose', 'hcount', 'mhtsd_', 'mhhigh', 'mhvwap'] or \
                i[:7] in ['hvolume', 'mhclose'] or i[:8] in ['mhvolume']:
            data['time_1']['feats'].append(i)
        elif i[:4] == 'mlow' or i[:5] in ['mtsd_', 'mhigh', 'mvwap', 'mmlow'] or \
                i[:6] in ['mclose', 'mmtsd_', 'mmhigh', 'mmvwap'] or \
                i[:7] in ['mvolume', 'mspread', 'mmclose'] or \
                i[:8] in ['mbid1vol', 'mask1vol', 'mmvolume', 'mmspread'] or \
                i[:9] in ['mmbid1vol', 'mmask1vol']:
            data['time_2']['feats'].append(i)
        elif i[:4] == 'slow' or i[:5] in ['stsd_', 'shigh', 'svwap', 'mslow'] or \
                i[:6] in ['sclose', 'mstsd_', 'mshigh', 'msvwap'] or \
                i[:7] in ['svolume', 'sspread', 'msclose'] or \
                i[:8] in ['sbid1vol', 'sask1vol', 'msvolume', 'msspread'] or \
                i[:9] in ['msbid1vol', 'msask1vol']:
            data['time_3']['feats'].append(i)
        elif i in static_feats:
            data['static_mins']['feats'].append(i)

    all_feats = ['symbol', 'label', 'entry_time', 'mtm_pl', 'zl1t']
    for k in data.keys():
        if 'time' in k:
            temp = list()
            for i in data[k]['feats']:
                an = len(re.findall('\d+', i)[-1])
                temp.append(i[:-an])
            data[k]['n_features'] = len(np.unique(temp))
        else:
            data[k]['n_features'] = len(data[k]['feats'])
        all_feats.extend(data[k]['feats'])
        print(k, data[k]['n_features'], 'features')
        if save_feats:
            file_path = '.'.join(save_feats.split('.')[:-1]) + '_' + k + '_features.npy'
            np.save(file_path, data[k]['feats'])
            print('Saved feats', k)

    return data, df[all_feats]


def getData(filename, sample=None, start_date=None, split_data=0.8, null_drop=False, cnn_transform=False,
            balance_classes=False, save_file=False, load_file=False, save_norms=False, load_norms=False):
    print('Getting', filename, dt.now())

    # load/return arrays
    if load_file:
        data = load_arrays(filename)
        if len(data):
            return data

    # read file, sample and add label
    df = readFile(filename, sample, start_date=start_date)
    df = add_label(df)
    print('Data Shape', df.shape)

    # get features
    feat_dict, df = get_features(df, save_feats=filename)
    print('Features Selected', df.shape)

    # drop rows with large null percentage and replace inf values
    if null_drop:
        df = drop_feats(df, feat_dict, null_drop=null_drop)
        print('Null Features Dropped', df.shape)

    # split data
    data = dict()
    df, df_test, data['y_train'], data['y_test'] = separate_balance_data(
        df, split_data, balance=balance_classes, save_test=filename
    )
    print('Data Separated and Classes Balanced', df.shape, df_test.shape)

    # format/scale data
    for fname in feat_dict.keys():
        ftype = fname.split('_')[0]
        feats = feat_dict[fname]['feats']
        scale = feat_dict[fname]['scale']
        time_steps = feat_dict[fname]['time_steps']
        n_features = feat_dict[fname]['n_features']
        clip_inputs = feat_dict[fname]['clip_inputs']
        reverse = feat_dict[fname]['reverse']
        if '.feather' in filename:
            norm_filename = filename.split('.feather')[0] + str(fname) + '.feather'
        else:
            norm_filename = filename.split('.csv')[0] + str(fname) + '.csv'
        data['X_train_' + str(fname)], data['X_test_' + str(fname)] = prep_matrix(
            df, df_test, feats, feature_type=ftype, time_steps=time_steps,
            n_features=n_features, scale=scale, clip_inputs=clip_inputs,
            reverse=reverse, save_norms=save_norms, load_norms=load_norms,
            filename=norm_filename
        )
        if cnn_transform and ftype == 'time':
            # data['X_train_rp' + str(fname)] = cnnTransform(copy.deepcopy(data['X_train_' + str(fname)]))
            data['X_test_rp' + str(fname)] = cnnTransform(copy.deepcopy(data['X_test_' + str(fname)]))
        print(fname, 'done')

    if save_file:
        save_arrays(data, filename)
        print('Arrays Saved')

    return data


def cnnTransform(X):
    T = X.shape[0]
    N = X.shape[1]
    D = X.shape[2]
    Z = np.zeros((N, T, T, D))
    for n in range(N):
        d_arr = np.zeros((T, T, D))
        for d in range(D):
            arr = X[:, n, d]
            d_arr[:, :, d] = rec_plot(arr)
        Z[n] = d_arr

    return Z


def rec_plot(ts, eps=0.001, steps=100, low_memory=False):
    if low_memory:
        d = pdist(ts[:, None])
        d = np.floor(d / eps)
        d[d > steps] = steps
        Z = squareform(d)
    else:
        N = ts.size
        Z = np.repeat(ts[None, :], N, axis=0)
        Z = np.floor(np.abs(Z - Z.T) / eps)
        Z[Z > steps] = steps

    return Z


def rnnTransform(X, time_steps, n_features, reverse=False):
    N = X.shape[0]
    T = int(time_steps)
    D = n_features
    data = np.zeros((T, N, D))
    ar = min([int(re.findall('\d+', i)[-1]) for i in X.columns])
    if reverse:
        rng = reversed(range(ar, time_steps + ar))
    else:
        rng = range(ar, time_steps + ar)
    for t in rng:
        t_step = np.zeros((N, D))
        n_f = 0
        for i in X.columns:
            an = int(re.findall('\d+', i)[-1])
            if t == an:
                t_step[:, n_f] = np.array(X[i]).astype(np.float32)
                n_f += 1
        data[t - ar] = t_step

    return data  # shape T, N, D


def prep_matrix(df, df_test, features, feature_type='static', scale='normalize',
                clip_inputs=True, fill_nans=True, time_steps=None, n_features=None,
                reverse=False, load_norms=False, save_norms=False, filename=None):
    print('Processing Data', dt.now())

    data = dict()
    if load_norms:
        data = load_arrays(filename, norm=True)
        if not data:
            load_norms = False
            print('!!!!!!!! No Data in Load Norms')

    X = df[features]
    X_test = df_test[features]

    if feature_type == 'time':
        print('Transforming Data to RNN Format', dt.now())
        X = rnnTransform(X, time_steps, n_features, reverse=reverse)
        X_test = rnnTransform(X_test, time_steps, n_features, reverse=reverse)
    else:
        X = np.array(X).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)

    if clip_inputs:
        if load_norms:
            minb = data['minb_norm']
            maxb = data['maxb_norm']
        else:
            if feature_type == 'time':  # assumes features are already normalized
                assert len(X.shape) == 3
                X_copy = X.reshape(-1, X.shape[-1])
                q1s = np.nanpercentile(X_copy, 25, axis=0)
                q3s = np.nanpercentile(X_copy, 75, axis=0)
            else:
                assert len(X.shape) == 2
                q1s = np.nanpercentile(X, 25, axis=0)
                q3s = np.nanpercentile(X, 75, axis=0)

            iqr = 1.5 * (q3s - q1s)
            minb = q1s - iqr
            maxb = q3s + iqr

        X = np.clip(X, minb, maxb)
        X_test = np.clip(X_test, minb, maxb)

        data['minb_norm'] = minb
        data['maxb_norm'] = maxb

    if fill_nans:
        if feature_type == 'time':  # assumes features are already normalized
            if load_norms:
                medians = data['nanmedians_norm']
            else:
                X_copy = X.reshape(-1, X.shape[-1])
                medians = np.nanmedian(X_copy, axis=0)
            for j in range(X.shape[1]):
                if not np.isnan(X[:, j]).sum():
                    continue
                for k in range(X.shape[2]):
                    if not np.isnan(X[:, j, k]).sum():
                        continue
                    for i in range(X.shape[0]):
                        v = X[i, j, k]
                        if np.isnan(v):
                            nulls = np.sum(np.isnan(X[:, j, k]))
                            if not nulls:
                                break
                            elif nulls == X.shape[0]:
                                X[:, j, k] = medians[k]
                                break
                            else:
                                if i != 0 and not np.isnan(X[i - 1, j, k]):
                                    X[i, j, k] = X[i - 1, j, k]
                                else:
                                    for i0 in range(i, X.shape[0]):
                                        if not np.isnan(X[i0, j, k]):
                                            X[i, j, k] = X[i0, j, k]
                                            break
            for j in range(X_test.shape[1]):
                if not np.isnan(X_test[:, j]).sum():
                    continue
                for k in range(X_test.shape[2]):
                    if not np.isnan(X_test[:, j, k]).sum():
                        continue
                    for i in range(X_test.shape[0]):
                        v = X_test[i, j, k]
                        if np.isnan(v):
                            nulls = np.sum(np.isnan(X_test[:, j, k]))
                            if not nulls:
                                break
                            elif nulls == X.shape[0]:
                                X_test[:, j, k] = medians[k]
                                break
                            else:
                                if i != 0 and not np.isnan(X_test[i - 1, j, k]):
                                    X_test[i, j, k] = X_test[i - 1, j, k]
                                else:
                                    for i0 in range(i, X_test.shape[0]):
                                        if not np.isnan(X_test[i0, j, k]):
                                            X_test[i, j, k] = X_test[i0, j, k]
                                            break
            if load_norms:
                zero_check = data['zero_check_norm']
            else:
                zero_check = np.nanmax(X_copy, axis=0) - np.nanmin(X_copy, axis=0)
                zero_check = np.argwhere(zero_check == 0)
            X = np.delete(X, zero_check, axis=2)
            X_test = np.delete(X_test, zero_check, axis=2)
        else:
            if load_norms:
                medians = data['nanmedians_norm']
            else:
                medians = np.nanmedian(X, axis=0)
            for i in range(X.shape[1]):
                if not np.isnan(X[:, i]).sum():
                    continue
                for j in range(X.shape[0]):
                    v = X[j, i]
                    if np.isnan(v):
                        X[j, i] = medians[i]
            for i in range(X_test.shape[1]):
                if not np.isnan(X_test[:, i]).sum():
                    continue
                for j in range(X_test.shape[0]):
                    v = X_test[j, i]
                    if np.isnan(v):
                        X_test[j, i] = medians[i]
            if load_norms:
                zero_check = data['zero_check_norm']
            else:
                zero_check = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
                zero_check = np.argwhere(zero_check == 0)
            X = np.delete(X, zero_check, axis=1)
            X_test = np.delete(X_test, zero_check, axis=1)

        data['nanmedians_norm'] = medians
        data['zero_check_norm'] = zero_check

    if scale:
        print('Scaling Data', dt.now())

        if load_norms:
            medians = data['medians_norm']
            q1s = data['q1s_norm']
            q3s = data['q3s_norm']
            means = data['means_norm']
            mins = data['mins_norm']
            maxs = data['maxs_norm']
            ranges = data['ranges_norm']
            qranges = data['qranges_norm']
            stds = data['stds_norm']
        else:
            if feature_type == 'time':
                # calculates normalization values based on each feature for all T and N combined
                assert len(X.shape) == 3
                X_copy = X.reshape(-1, X.shape[-1])
                medians = np.nanmedian(X_copy, axis=0)
                q1s = np.nanpercentile(X_copy, 25, axis=0)
                q3s = np.nanpercentile(X_copy, 75, axis=0)
                means = np.nanmean(X_copy, axis=0)
                mins = np.nanmin(X_copy, axis=0)
                maxs = np.nanmax(X_copy, axis=0)
                ranges = maxs - mins + 10e-10
                qranges = q3s - q1s + 10e-10
                stds = np.nanstd(X_copy, axis=0) + 10e-10

            else:
                assert len(X.shape) == 2
                medians = np.nanmedian(X, axis=0)
                q1s = np.nanpercentile(X, 25, axis=0)
                q3s = np.nanpercentile(X, 75, axis=0)
                means = np.nanmean(X, axis=0)
                mins = np.nanmin(X, axis=0)
                maxs = np.nanmax(X, axis=0)
                ranges = maxs - mins + 10e-10
                qranges = q3s - q1s + 10e-10
                stds = np.nanstd(X, axis=0) + 10e-10

        data['medians_norm'] = medians
        data['q1s_norm'] = q1s
        data['q3s_norm'] = q3s
        data['means_norm'] = means
        data['mins_norm'] = mins
        data['maxs_norm'] = maxs
        data['ranges_norm'] = ranges
        data['qranges_norm'] = qranges
        data['stds_norm'] = stds

        if scale == 'medianscale':
            X -= medians
            X /= stds
            X_test -= medians
            X_test /= stds
        elif scale == 'meanscale':
            X -= means
            X /= stds
            X_test -= means
            X_test /= stds
        elif scale == 'normalize':
            X -= mins
            X /= ranges
            X_test -= mins
            X_test /= ranges
        elif scale == 'prcntchange':
            Z = np.zeros(X.shape)[:-1]
            for i in range(Z.shape[0]):
                Z[i] = (X[i + 1] - X[i]) / X[i]
            X = Z
            Z = np.zeros(X_test.shape)[:-1]
            for i in range(Z.shape[0]):
                Z[i] = (X_test[i + 1] - X_test[i]) / X_test[i]
            X_test = Z
        elif scale == 'prcntrngchange':
            Z = np.zeros(X.shape)[:-1]
            for i in range(Z.shape[0]):
                Z[i] = (X[i + 1] - X[i]) / ranges
            X = Z
            Z = np.zeros(X_test.shape)[:-1]
            for i in range(Z.shape[0]):
                Z[i] = (X_test[i + 1] - X_test[i]) / ranges
            X_test = Z

    n_check0 = len(np.argwhere(np.isnan(X)))
    n_check1 = len(np.argwhere(np.isinf(X)))
    n_check2 = len(np.argwhere(np.isneginf(X)))
    if n_check0 or n_check1 or n_check2:
        print('!!!!!!!! Null/Inf Values Remaining in X',
              max(n_check0, n_check1, n_check2) / float(len(X.flatten())))
        X = np.nan_to_num(X)
    n_check0 = len(np.argwhere(np.isnan(X_test)))
    n_check1 = len(np.argwhere(np.isinf(X_test)))
    n_check2 = len(np.argwhere(np.isneginf(X_test)))
    if n_check0 or n_check1 or n_check2:
        print('!!!!!!!! Null/Inf Values Remaining in X_test',
              max(n_check0, n_check1, n_check2) / float(len(X_test.flatten())))
        X_test = np.nan_to_num(X_test)

    if save_norms:
        save_arrays(data, filename)

    return X, X_test


def get_ml_type(df, label):
    if type(df) == str:
        df = pd.read_feather(df)
    if len(df[label].unique()) <= 10:
        return 'clf'
    else:
        return 'reg'


def stagesData(final=False):
    stages = dict()
    stages['stage1'] = dict()
    stages['stage2'] = dict()
    stages['stage3'] = dict()
    stages['stage4'] = dict()
    stages['merge_best'] = 3
    stages['stage1']['num_feats'] = 2500
    stages['stage2']['num_feats'] = 1000
    stages['stage3']['num_feats'] = 500
    stages['stage4']['num_feats'] = 250
    stages['stage1']['batch_sz'] = 50
    stages['stage2']['batch_sz'] = 100
    stages['stage1']['num_rounds'] = 50
    stages['stage2']['num_rounds'] = 100
    stages['stage3']['num_rounds'] = 200
    stages['stage4']['num_rounds'] = 400
    stages['stage1']['max_depth'] = 1
    stages['stage2']['max_depth'] = 2
    stages['stage3']['max_depth'] = 4
    stages['stage4']['max_depth'] = 6
    stages['stage1']['num_leaves'] = 2
    stages['stage2']['num_leaves'] = 4
    stages['stage3']['num_leaves'] = 6
    stages['stage4']['num_leaves'] = 8
    stages['stage1']['learning_rate'] = 0.08
    stages['stage2']['learning_rate'] = 0.04
    stages['stage3']['learning_rate'] = 0.02
    stages['stage4']['learning_rate'] = 0.01
    stages['stage1']['filter_th'] = 0.6
    stages['stage2']['filter_th'] = 0.7
    stages['stage3']['filter_th'] = 0.8
    stages['stage4']['filter_th'] = 0.9

    if final:
        stages['stage1']['filter_th'] = 0.8
        stages['stage2']['filter_th'] = 0.85
        stages['stage3']['filter_th'] = 0.9
        stages['stage4']['filter_th'] = 0.95
        stages['stage4']['num_feats'] = 100
        stages['stage4']['max_depth'] = 8
        stages['stage4']['num_leaves'] = 16

    return stages


def paramData(nn_params=False):
    if not nn_params:
        param_ranges = dict(range_params=dict(), feat_params=dict(), fixed_params=dict())
        param_ranges['range_params']['max_depth'] = list(range(6, 18, 2))
        param_ranges['range_params']['num_leaves'] = list(range(4, 102, 12))
        param_ranges['range_params']['learning_rate'] = [0.005, 0.01, 0.02, 0.04]
        param_ranges['range_params']['min_data_in_leaf'] = list(range(500, 3500, 500))
        param_ranges['range_params']['num_round'] = list(range(100, 800, 200))
        param_ranges['range_params']['year_lookback'] = list(range(1, 4, 1))
        param_ranges['feat_params']['feat_num'] = [50, 75]
        param_ranges['feat_params']['model_type'] = ['tree']
        param_ranges['fixed_params']['metric'] = ['binary_logloss', 'auc', 'l1', 'l2']

    else:
        import tensorflow as tf
        from tensorflow.contrib.rnn import GRUCell, LSTMCell, LayerNormBasicLSTMCell

        param_ranges = dict(
            fixed_params=dict(), cost_params=dict(), cell_params=dict(),
            fc_layer_params=dict(), rnn0_layer_params=dict(),
            rnn1_layer_params=dict(), final_layer_params=dict()
        )
        param_ranges['fixed_params']['max_seconds'] = [600]
        param_ranges['fixed_params']['epochs'] = [1000]
        param_ranges['fixed_params']['learning_rate'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        param_ranges['fixed_params']['batch_norm_decay'] = [0.8, 0.9, 0.99]
        param_ranges['fixed_params']['fc_keep_probs'] = [0.6, 0.8, None]
        param_ranges['fixed_params']['rnn0_keep_prob'] = [0.6, 0.8, 1.0]
        param_ranges['fixed_params']['rnn1_keep_prob'] = [0.6, 0.8, 1.0]
        param_ranges['fixed_params']['final_keep_probs'] = [0.6, 0.8, None]
        param_ranges['fixed_params']['activation_func'] = [tf.nn.elu, tf.nn.relu]
        param_ranges['fixed_params']['rnn_activation_func'] = [tf.nn.elu, tf.nn.relu, tf.tanh]
        param_ranges['fixed_params']['batch_sz'] = [256, 512, 1024]
        param_ranges['cost_params']['cost_func'] = [tf.train.AdamOptimizer]
        param_ranges['cost_params']['beta1'] = [0.9, 0.99, 0.999]
        param_ranges['cost_params']['beta2'] = [0.9, 0.99, 0.999]
        param_ranges['cost_params']['epsilon'] = [1e-2, 1e-8]
        param_ranges['cell_params']['rnn_cell'] = [LSTMCell, LayerNormBasicLSTMCell, GRUCell]
        param_ranges['cell_params']['use_peepholes'] = [True, False]
        param_ranges['fc_layer_params']['fc_layers'] = list(range(1, 4))
        param_ranges['fc_layer_params']['fc_neurons'] = list(range(20, 220, 20))
        param_ranges['rnn0_layer_params']['rnn0_layers'] = list(range(1, 4))
        param_ranges['rnn0_layer_params']['rnn0_neurons'] = list(range(5, 55, 5))
        param_ranges['rnn1_layer_params']['rnn1_layers'] = list(range(1, 4))
        param_ranges['rnn1_layer_params']['rnn1_neurons'] = list(range(5, 55, 5))
        # param_ranges['rnn2_layer_params']['rnn2_layers'] = list(range(1, 4))
        # param_ranges['rnn2_layer_params']['rnn2_neurons'] = list(range(5, 55, 5))
        # param_ranges['rnn3_layer_params']['rnn3_layers'] = list(range(1, 4))
        # param_ranges['rnn3_layer_params']['rnn3_neurons'] = list(range(5, 55, 5))
        param_ranges['final_layer_params']['final_layers'] = list(range(1, 4))
        param_ranges['final_layer_params']['final_neurons'] = list(range(10, 110, 10))

    return param_ranges