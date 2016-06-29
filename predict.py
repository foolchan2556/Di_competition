import numpy as np
import pandas as pd
import xgboost as xgb
import data_helper as dp

def complete(infile,outfile):
    mdict = {}
    rows = open(infile).readlines()
    for row in rows:
        (src,time,label) = row.split(',')
        mdict[src+time] = label
    res = []
    for date in ['23','25','27','29','31']:
        for slc in range(46,144,12):
            for src in range(1,67):
                stamp = '2016-01-'+date+'-'+str(slc)
                mkey = str(src)+ stamp
                if mkey in mdict and float(mdict[mkey])>1.0:
                    res.append(str(src)+','+stamp+','+mdict[mkey])
                else:
                    res.append(str(src)+','+stamp+',1.0\n')
    open(outfile,'w').writelines(res)

train = pd.read_csv('../data/train_full_dummies.csv')
test = pd.read_csv('../data/test_full_dummies.csv')

train_labels = train['gap']
train_features = train.drop(['gap','Weather','Weekday','src','label_time_slice','date'], axis=1)
train_label_array = np.array(train_labels.tolist())

test_labels = test['gap']
test_features = test.drop(['gap','Weather','Weekday','src','label_time_slice','date'], axis=1)
test_label_array = np.array(test_labels.tolist())

def custobj(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = np.zeros(y_true.shape)
    hess = np.zeros(y_pred.shape)
    y = y_true[y_true != 0]
    yh = y_pred[y_true != 0]
    grad[y_true != 0] = (yh-y)/y ** 2
    hess[y_true != 0] = 1. / y ** 2
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds[labels > 0]
    labels = labels[labels > 0]
    return 'MAPE', 0.636*np.mean(np.abs(labels - preds)/labels)

params = {"objective": "reg:linear",
          "max_depth": 6,
          "eta": 0.1,
          "gamma": 0,
          "min_child_weight": 4,
          "silent": 0,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "nthread": 16,
          "seed": 27}
num_trees=300

gbm = xgb.train(params, xgb.DMatrix(train_features, train_labels), num_trees, obj=custobj, feval=evalerror)

predicted_labels = gbm.predict(xgb.DMatrix(test_features))
predicted_label_array = np.maximum(predicted_labels, np.ones(predicted_labels.shape))

test['label'] = predicted_label_array.tolist()
test['timestamp'] = test.apply(lambda x: x['date']+'-'+ str(x['label_time_slice']),axis=1)
res_df = pd.concat([test['src'],test['timestamp'],test['label']],axis=1)
pfix = '3'
res_df.to_csv('../data/res_'+pfix+'.csv', index=False)
complete('../data/res_'+pfix+'.csv','../data/res_'+pfix+'_1.csv')
