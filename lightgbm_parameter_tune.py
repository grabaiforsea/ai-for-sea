import numpy as np
import pandas as pd 

from lightgbm import LGBMClassifier
from hyperopt import hp, tpe
from hyperopt.fmin import fmin


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#   Tuning with bayesian optimization with hyperopt
#   Tuning with 1/8 of training data for efficiency.

sampled_train = pd.read_csv('train_full.csv').sample(frac = 0.125 , random_state = 100)
sampled_train.drop('bookingID' ,axis = 1 , inplace = True)
sampled_train.reset_index(drop= True , inplace = True)

measurements = [
    'acceleration',
    'gyro'
]

norm_pairs = [
    'xy',
    'xz',
    'yz'
]

measurements = [
    'acceleration',
    'gyro'
]

norm_pairs = [
    'xy',
    'xz',
    'yz'
]

for m in measurements :
    sampled_train['{}_norm'.format(m)] = np.linalg.norm(sampled_train.filter(regex = '{}_[xyz]'.format(m)), 
                                                        axis = 1)

    for pair in norm_pairs :
        sampled_train['{}_norm_{}'.format(m,pair)] = np.linalg.norm(sampled_train.filter(regex='{}_[{}]'.format(m,pair)),
                                                                  axis = 1)
sampled_train['Speed'].replace(-1 , np.nan , inplace = True)

N_SPLITS = 4

kf = StratifiedKFold(N_SPLITS , random_state = 42)

X = sampled_train.loc[:,sampled_train.columns != 'labels'].values
y = sampled_train['labels'].values

space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
    'num_leaves': hp.quniform('num_leaves', 50, 200, 2),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
}

def cross_val_score(params) :
    kf_score = 0
    int_params = ['num_leaves','subsample_for_bin','min_child_samples']
    for param in int_params :
        params[param] = int(params[param])
    print(params)
    print()
    for i , (train_idx, val_idx) in enumerate(kf.split(X,y)) :
        model = LGBMClassifier(n_estimators = 750 , n_jobs = -1, random_state = 100 , **params)
        X_train, X_val , y_train , y_val = X[train_idx] , X[val_idx] , y[train_idx] , y[val_idx]
        model.fit(X_train , y_train)
        y_pred = model.predict_proba(X_val)[:,1]
        fold_score = roc_auc_score(y_val , y_pred)
        kf_score += fold_score/N_SPLITS
    loss = 1-kf_score
    print('LOSS :', loss)
    return loss
    
opt_res = fmin(fn = cross_val_score , space = space , max_evals=40 , algo=tpe.suggest)
print(opt_res)