import numpy as np
import pandas as pd

#from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import gc

from warnings import simplefilter
import os
import sys


simplefilter('ignore')

print(os.listdir("../input"))

N_SPLITS = 4

speed_idx = 9

X = np.load('../input/train.npy')
y = np.load('../input/labels.npy')

meta_train = pd.DataFrame(np.zeros(len(X)) , columns=['preds'])

#X_test = test_data.values

kf = StratifiedKFold(N_SPLITS, random_state=42)

kf_score = 0
n_features = X.shape[1]
impute_slice = slice(speed_idx,speed_idx+1)

for i , (train_idx, val_idx) in enumerate(kf.split(X,y)) :
    if i > 1 :
        continue
    X_train, X_val , y_train , y_val = X[train_idx] , X[val_idx] , y[train_idx] , y[val_idx]
    impute = SimpleImputer(-1, strategy = 'median')
    X_train[:,impute_slice] = impute.fit_transform(X_train[:,impute_slice])
    X_val[:,impute_slice] = impute.transform(X_val[:,impute_slice])
    #model = AdaBoostClassifier(ExtraTreesClassifier(20,n_jobs = -1) , n_estimators = 50)
    model = AdaBoostClassifier(ExtraTreesClassifier(10, n_jobs = -1) , n_estimators = 400)
    model.fit(X_train , y_train)
    
    y_pred_train = model.predict_proba(X_train)[:,1]
    y_pred = model.predict_proba(X_val)[:,1]
    #y_pred_test = model.predict(X_t)[:,1]
    meta_train.iloc[val_idx,0] = y_pred
    #meta_test.iloc[:,0] = y_pred_test/N_SPLITS
    
    fold_train_score = roc_auc_score(y_train , y_pred_train)
    print('Fold', i+1, 'Train Score : ', fold_train_score)
    
    fold_score = roc_auc_score(y_val , y_pred)
    print('Fold' , i+1, 'val score :', fold_score)
    kf_score += fold_score/N_SPLITS
    print()
    
    del X_train, X_val , y_train , y_val
    gc.collect()
    
print('CV-Score :',kf_score)

meta_train.to_csv('meta_train.csv', index = False)