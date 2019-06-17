import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import gc

from warnings import simplefilter
import os
import sys

simplefilter('ignore')

N_SPLITS = 4

speed_idx = 9 #refer to base file

X = np.load('train.npy')
#X_test = np.load('test.npy')
y = np.load('labels.npy')

#meta_train = pd.DataFrame(np.zeros(len(X)) , columns=['preds'])
#meta_test = np.zeros(len(X_test))

kf = StratifiedKFold(N_SPLITS, random_state=42)

kf_score = 0
n_features = X.shape[1]
impute_slice = slice(speed_idx,speed_idx+1)

#X_test = test_data.values

kf = StratifiedKFold(N_SPLITS, random_state=42)

#del test_data
gc.collect()

def roc_eval(y_preds ,train_data ) :
    y_true = train_data.label
    return ('roc-score', roc_auc_score(y_true, y_preds), True)

kf_score = 0
n_features = X.shape[1]

train_params = {
    #'colsample_by_tree' : 0.81525,
    'is_unbalance' : True,
    'learning_rate' : 0.0944,
    'min_child_samples' : 295,
    'num_leaves' : 196,
    'reg_alpha' : 0.93262,
    'reg_lambda' : 0.42029,
    'subsample_for_bin' : 200000,
    'random_state' : 42,
    'n_jobs' : -1,
    'objective' : 'binary',
    'metric' : 'None'
}

gc.collect()

for i , (train_idx, val_idx) in enumerate(kf.split(X,y)) :
    print('Fold' , i+1 , '  Training Starts')
    lgb_train = lgb.Dataset(X[train_idx], y[train_idx])
    lgb_valid = lgb.Dataset(X[val_idx] , y[val_idx], reference = lgb_train)
    bst = lgb.train(train_params , lgb_train , 9000, valid_sets=lgb_valid,feval = roc_eval, early_stopping_rounds = 200, verbose_eval = 500)
    bst.save_model('model_{}.txt'.format(i))