import os
import re
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.signal import periodogram
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest


def get_filenames():
    filenames=sorted(os.listdir("DATA_CSV"))
    return filenames

def get_data_from_filename(filename):
    data=pd.read_csv(f'DATA_CSV/{filename}', index_col=0)
    data=data[::6] #real sampling rate
    print (f"Getting data from file {filename} with shape {data.shape}")
    return data.to_numpy()

def get_filenames_specific_data(statuses, amplitudes):
    filenames=get_filenames()
    filenames_status=[]
    filenames_amplitude=[]
    for status in statuses:
        filenames_status.extend(filenames_with_status(status))
    for amplitude in amplitudes:
        filenames_amplitude.extend(filenames_with_amplitude(amplitude))
    return [element for element in filenames_status if element in filenames_amplitude]

def get_specific_data(statuses,amplitudes):
    filenames=get_filenames_specific_data(statuses, amplitudes)
    data=[]
    print(filenames)
    for filename in filenames:
        data.append(get_data_from_filename(filename))

    data=append_data_from_nparray(data)
    return data
        
def filenames_with_amplitude(amplitude):
    amplitude_filenames=[]
    filenames=get_filenames()
    for filename in filenames:
        s=re.search("(?<=[0-9]\_[0-9]\_)[0-9]{1,2}(?=A.csv)", filename)
        if s==None:
            s=re.search("(?<=[0-9]\_[0-9]{2}\_)[0-9]{1,2}(?=A.csv)", filename)
        if int(s.group(0))==amplitude:
            amplitude_filenames.append(filename)
    return amplitude_filenames

def filenames_with_status(status):
    filenames=get_filenames()
    filenames = [x for x in filenames if int(x[0])==status]
    return filenames


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def scale (scaler,data):
    scaled_data=scaler.transform(data)
    return scaled_data

def get_scaler_total_healthy():
    data = get_specific_data(statuses=[1],amplitudes=[1,2,3,5])
    data=reshape(data,199)
    scaler=fit_scaler(data)
    data=scale(scaler, data)

    return scaler



def get_healthy_matrixes():
    data = get_specific_data(statuses=[1],amplitudes=[1,2,3,5])

    train, test=train_test_split(data, test_size=0.2)

    X_train=reshape(train,199)
    X_test=reshape(test,199)

    scaler=fit_scaler(X_train)
    X_train = scale (scaler,X_train)
    X_test = scale (scaler,X_test)

    return X_train, X_test, scaler

#this is necessary so we can construct a v of amps since otherwise we shuffle the data
def get_healthy_matrixes_no_split(scaler):
    data = get_specific_data(statuses=[1],amplitudes=[1,2,3,5])

    X_data=reshape(data,199)

    X_data = scale (scaler,X_data)

    v_of_statuses=len(X_data)*[1]

    return X_data, v_of_statuses

def append_data_from_nparray(nparray_list):
    concatenated_data=np.concatenate(tuple(nparray_list))
    return concatenated_data

def reshape (data, k):
    reshape=[]
    data=data.T
    i=0
    while i < len(data[0]):
        sample=[]
        for sensor in range(0,24):
            s=data[sensor][i:i+k]
            sample=np.append(sample,s)
        reshape.append(sample)
        i=i+k
    reshape = np.array(reshape)
    print (f"Reshaping []data with shape {data.shape} to {reshape.shape}, with k={k}, len(data[0])/k={len(data[0])}/{k}={len(data[0])/k} samples")
    return reshape


def get_replica_matrixes(scaler):
    other_matrixes=[]
    v_of_statuses, v_of_amplitudes=[], []
    for amplitude in [1,2,3,5]:
            data=get_specific_data(statuses=[2], amplitudes=[amplitude])
            data=reshape (data, 199)
            data=scale (scaler,data)
            v_of_statuses+=[2]*len(data)
            v_of_amplitudes+=[amplitude]*len(data)
            other_matrixes.append(data)
    data=append_data_from_nparray(other_matrixes)
    return data, v_of_statuses, v_of_amplitudes
    

def get_other_matrixes(scaler):
    other_matrixes=[]
    v_of_statuses, v_of_amplitudes=[], []
    for status in range(3,5):
        for amplitude in [1,2,3,5]:
            data=get_specific_data(statuses=[status], amplitudes=[amplitude])
            data=reshape (data, 199)
            data=scale (scaler,data)
            v_of_statuses+=[status]*len(data)
            v_of_amplitudes+=[amplitude]*len(data)
            other_matrixes.append(data)
    data= append_data_from_nparray(other_matrixes)
    return data, v_of_statuses, v_of_amplitudes

def get_replica_matrixes(scaler):
    other_matrixes=[]
    v_of_statuses, v_of_amplitudes=[], []
    for amplitude in [1,2,3,5]:
        data=get_specific_data(statuses=[2], amplitudes=[amplitude])
        data=reshape (data, 199)
        data=scale (scaler,data)
        v_of_statuses+=[2]*len(data)
        v_of_amplitudes+=[amplitude]*len(data)
        other_matrixes.append(data)
    data= append_data_from_nparray(other_matrixes)
    return data, v_of_statuses, v_of_amplitudes

def rms(x):
    rms=np.mean([number ** 2 for number in x])**.5
    return rms

def pp(x):
    pp=max(x)-min(x)
    return pp

def zp(x):
    zp=pp(x)/2
    return zp

def cf(x):
    cf=zp(x)/rms(x)
    return cf

def _xlogx(x, base=2):
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    return np.where(x == 0, 0, x * np.log(x) / np.log(base))


def entropy(x):
    _, psd = periodogram(x)
    psd_norm = psd / psd.sum(keepdims=True)
    se = -_xlogx(psd_norm).sum()
    return se

def get_features(data, params=None):

    params_dict = {"mean":np.mean, "std":np.std, "kurt":kurtosis,"rms":rms, "pp":pp, "zp":zp, "cf":cf, "ent":entropy, }
    features=[]

    if params==None:
        params=params_dict.keys()
   
    for sample in data:
        sample_feats=[]
        for param in params:
            sample_feats.append(params_dict[param](sample))
        features.append(sample_feats)

    return features

def get_features_from_filename(filename):
    pattern= "(.*)(?=A\.csv)"
    m= re.match(pattern,filename)
    l=tuple(map(int, m.group(0).split("_"))) #refactor que està lleig
    return l #(status, sample, amplitude)


def get_data_with_amp (scaler):
    matrixes=[]
    v_of_amps=[]
    v_of_status=[]
    for amplitude in [1,2,3,5]:
        for status in [1,2,3,4]:
            data=get_specific_data(statuses=[status], amplitudes=[amplitude])
            data=scale (scaler,data)
            data=reshape (data, 199)
            v_of_amps+=[amplitude]*len(data)
            v_of_status+=[status]*len(data)
            matrixes.append(data)
    data= append_data_from_nparray(matrixes)
    return data, v_of_amps, v_of_status


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots()
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

def train_predict_SVM(df_train,dict_for_test,dict_test_expected,params):
    clf = OneClassSVM(kernel='sigmoid', nu=0.05).fit(df_train[params])
    y_true=[]
    y_pred=[]
    for dataset in dict_for_test:
        res=clf.predict(dict_for_test[dataset][params])
        print(f"{dataset}:{np.unique(res,return_counts=True)}")
        y_true.extend([dict_test_expected[dataset]]*len(res))
        y_pred.extend(res)
    return plot_cm(y_true, y_pred)


def train_predict_SVM_noshow(train,dict_for_test,dict_test_expected,params, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5):
    print(f"OneClassSVM {params} kernel={kernel}, nu = {nu}")
    clf = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu).fit(train[params])
    y_true=[]
    y_pred=[]
    results={}
    for dataset in dict_for_test:
        res=clf.predict(dict_for_test[dataset][params])
        results[dataset]=np.unique(res,return_counts=True)
        y_true.extend([dict_test_expected[dataset]]*len(res))
        y_pred.extend(res)
        cm=confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        accuracy=accuracy_score(y_true, y_pred)
    return results, cm, accuracy

def train_predict_Isolation_noshow(train,dict_for_test,dict_test_expected,params, random_outliers , n_estimators=100):
    print(f"Isolation forest {params}, n_estimators = {n_estimators}")
    train_data=append_data_from_nparray([train[params],random_outliers])
    clf = IsolationForest(random_state=34,n_estimators=n_estimators,contamination=len(random_outliers)/len(train_data)).fit(train_data)
    y_true=[]
    y_pred=[]
    results={}
    for dataset in dict_for_test:
        res=clf.predict(dict_for_test[dataset][params])
        results[dataset]=np.unique(res,return_counts=True)
        y_true.extend([dict_test_expected[dataset]]*len(res))
        y_pred.extend(res)
        cm=confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        accuracy=accuracy_score(y_true, y_pred)
    return results, cm, accuracy

def plot_cm2(cf_matrix):  
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
    return