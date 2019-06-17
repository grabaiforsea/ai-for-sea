import numpy as np
import pandas as pd

import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 4

speed_idx = 9 #refer to base file

train_data = pd.read_csv('train_full.csv' , usecols= [i+1 for i in range(11)])
#test_data = pd.read_csv('test_full.csv', usecols= [i+1 for i in range(11)])

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
    train_data['{}_norm'.format(m)] = np.linalg.norm(train_data.filter(regex = '{}_[xyz]'.format(m)), 
                                                        axis = 1)
    #test_data['{}_norm'.format(m)] = np.linalg.norm(test_data.filter(regex = '{}_[xyz]'.format(m)), 
    #                                                    axis = 1)
    for pair in norm_pairs :
        train_data['{}_norm_{}'.format(m,pair)] = np.linalg.norm(train_data.filter(regex='{}_[{}]'.format(m,pair)),
                                                                  axis = 1)
    #    test_data['{}_norm_{}'.format(m,pair)] = np.linalg.norm(test_data.filter(regex='{}_[{}]'.format(m,pair)),
    #                                                              axis = 1)

train_data['Speed'].replace(-1 , np.nan , inplace = True)
#test_data['Speed'].replace(-1, np.nan , inplace = True)

meta_train = pd.DataFrame(np.zeros(len(train_data)) , columns=['preds'])
meta_test = pd.DataFrame(np.zeros(len(test_data)) , columns=['preds'])

X = train_data.loc[:,train_data.columns != 'labels'].values
y = train_data['labels'].values
#X_test = test_data.values

meta_train = pd.DataFrame(np.zeros(len(X)) , columns=['preds'])
#meta_test = np.zeros(len(X_test))

kf = StratifiedKFold(N_SPLITS, random_state=42)

del train_data
#del test_data
gc.collect()

def run_inference_train(i) :
    bst = lgb.Booster(model_file = '../input/lgbm-fold-{}-part-1/model_{}.txt'.format(i+1,i))
    return bst.predict(X[idxs[i]]) , bst.predict(X_test)

idxs = [val_idx for train_idx, val_idx in kf.split(X,y)]

with Pool(os.cpu_count()) as p :
    preds_list =  p.map(run_inference_train , range(4))

for i , val_idx in enumerate(idxs) :
    meta_train.iloc[val_idx,0] = preds_list[i][0]
    meta_test += preds_list[i][1]

np.save('meta_test', meta_test)
meta_train.to_csv('meta_train.csv')