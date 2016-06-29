import numpy as np
import pandas as pd
import xgboost as xgb
import data_helper as dp

# train = pd.read_csv('../data/train_full.csv').drop(['label', 'Unnamed: 0'], axis=1)
# valid = pd.read_csv('../data/train_full.csv').drop(['label', 'Unnamed: 0'], axis=1)
train = pd.read_csv('../data/train_full_dummies.csv').drop(['date', 'Unnamed: 0'], axis=1)

train_labels = train['gap']
train_features = train.drop(['gap'], axis=1)
#train_features['Temperature'] = np.ones(train_features.shape)*26-  train_features['Temperature']

def custobj(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = np.zeros(y_true.shape)
    hess = np.zeros(y_pred.shape)
    y = y_true[y_true != 0]
    yh = y_pred[y_true != 0]
    grad[y_true != 0] = (yh-y)/y ** 2
    hess[y_true != 0] = 1. / y ** 2
    return grad, hess

# xgboost.cv(params, dtrain, num_boost_round=10, nfold=3,
# stratified=False, folds=None, metrics=(), obj=None, feval=None,
# maximize=False, early_stopping_rounds=None, fpreproc=None,
# as_pandas=True, verbose_eval=None, show_stdv=True, seed=0, callbacks=None)

params_cv = {
           "max_depth": 7,
          "learning_rate": 0.2,
          "gamma": 2,
          "min_child_weight": 1,
          "silent": 0,
          "subsample": 0.9,
          "colsample_bytree": 0.9,
          "nthread": 36,
          "seed": 27}

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    #avgs = dtrain.get_label()[:,1]
    error = np.zeros(labels.shape)
    #preds = (preds if preds > avgs else avgs*2)[labels>0]
    preds = preds[labels>0]
    labels = labels[labels > 0]
    error[labels > 0] = np.abs(labels - preds)/labels
    return 'MAPE', np.mean(error)

'''
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    error = np.zeros(labels.shape)
    preds = preds[labels > 0]
    labels = labels[labels > 0]
    error[labels > 0] = np.abs(labels - preds)/labels
    return 'MAPE', np.mean(error)
'''
result = xgb.cv(params_cv, xgb.DMatrix(train_features, train_labels),
       num_boost_round=800, nfold=5,
       obj=custobj, feval=evalerror)
print result
