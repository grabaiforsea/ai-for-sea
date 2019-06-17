import glob
import pandas as pd
import numpy as np


########################################################

label_dict = pd.read_csv('./labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv').set_index('bookingID').to_dict()['label']

headers = pd.read_csv('./features/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv').columns.tolist() + ['labels']
train_data = pd.DataFrame(columns = headers)

for f in glob.glob('./features/*.csv') :
    temp_df = pd.read_csv(f)
    temp_df['labels'] = temp_df['bookingID'].map(label_dict)
    train_data = train_data.append(temp_df)

train_data.to_csv('train_full.csv' , index = False)

#########################################################

#### In case combined train data already exists, please rename it as train_full.csv and comment the section above

#train_data = pd.read_csv('train_full.csv') #Please uncomment in case full train data already available
#test_data = pd.read_csv('test_full.csv')

train_data.drop('bookingID', axis = 1, inplace = True)
test_data.drop('bookingID', axis = 1 , inplace = True)

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
    # test_data['{}_norm'.format(m)] = np.linalg.norm(test_data.filter(regex = '{}_[xyz]'.format(m)), 
    #                                                    axis = 1)
    for pair in norm_pairs :
        train_data['{}_norm_{}'.format(m,pair)] = np.linalg.norm(train_data.filter(regex='{}_[{}]'.format(m,pair)),
                                                                  axis = 1)

        #test_data['{}_norm_{}'.format(m,pair)] = np.linalg.norm(test_data.filter(regex='{}_[{}]'.format(m,pair)),
        #                  

X = train_data.loc[:,train_data.columns != 'labels'].values
y = train_data['labels'].values
X_test = test_data.values

np.save('train.npy', X)
np.save('test.npy', X_test)
np.save('labels.npy', y)