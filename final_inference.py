import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

#### Uncomment for test data inference
#test_data = pd.read_csv('test_full.csv')[['bookingData']]
#test_data['preds'] = np.load('meta_test.npy')
#test_data.groupby('bookingID').agg({'preds' : 'mean'})
#test_data.to_csv('prediction.csv', index = False)

train_data = pd.read_csv('train_full.csv')[['bookingData','labels']]
train_data['preds'] = pd.read_csv('meta_train.csv').values
grp = train_data.groupby('bookingID').agg({'labels' : 'max' , 'preds' : 'mean'})
print(roc_auc_score(grp['labels'] , grp['preds']))