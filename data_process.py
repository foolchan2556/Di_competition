import numpy as np
import pandas as pd
import xgboost as xgb
import data_helper as dp

# generate csv of training dataset and testing dataset by date
print("Generating daily training data csv and testing data csv")
train_full = dp.merge_df('../data/training_data/',['2016-01-0'+str(i+1) for i in range(9)]+['2016-01-'+str(i) for i in range(10,22)])
test = dp.test_df('../data/test_set_2/',['2016-01-23','2016-01-25','2016-01-27','2016-01-29','2016-01-31'])

# process testing dataset
print("Loading the training/test data using pandas")
train_list = []
test_list = []
for tf in ['2016-01-0'+str(i) for i in range(2,10)]+['2016-01-1'+str(i) for i in range(10)]+['2016-01-20','2016-01-21']:
    temp = pd.read_csv('train_'+tf+'.csv').drop(['label', 'Unnamed: 0'], axis=1)
    temp['date'] = [tf for i in range(len(temp))]
    train_list.append(temp)
train = pd.concat(train_list)
for tf in ['2016-01-23','2016-01-25','2016-01-27','2016-01-29','2016-01-31']:
    temp = pd.read_csv('test_'+tf+'.csv').drop(['label', 'Unnamed: 0'], axis=1)
    temp['date'] = [tf for i in range(len(temp))]
    test_list.append(temp)
test = pd.concat(test_list)
data = pd.concat([train,test])
for column in ['Weather', 'Weekday', 'label_time_slice', 'src']:
    data[column] = data[column].astype(object)
    dummies = pd.get_dummies(data[column], prefix=column)
    data[dummies.columns] = dummies
train_dum = data[:len(train)]
test_dum = data[len(train):]
train_dum.to_csv('train_full_dummies.csv', index=False)
test_dum.to_csv('test_full_dummies.csv', index=False)
