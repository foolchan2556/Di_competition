import numpy as np
import pandas as pd
import xgboost as xgb
import data_prepare_nodst_nodeal as dp
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston

train = pd.read_csv('../data/train_with_dummies.csv').drop(['Weekday'], axis=1)
valid = pd.read_csv('../data/valid_with_dummies.csv').drop(['Weekday'], axis=1)

train_labels = train['gap']
train_features = train.drop(['gap'], axis=1)
train_label_array = np.array(train_labels.tolist())

valid_labels = valid['gap']
valid_features = valid.drop(['gap'], axis=1)
valid_label_array = np.array(valid_labels.tolist())


def custobj(y_true, y_pred):
    grad = np.zeros(y_true.shape)
    hess = np.zeros(y_pred.shape)
    y = y_true[y_true != 0]
    yh = y_pred[y_true != 0]
    grad[y_true != 0] = (yh-y)/y ** 2
    hess[y_true != 0] = 1. / y ** 2
    return grad, hess

params = {"objective": custobj,
          "max_depth": 6,
          "learning_rate": 0.5,
          "gamma": 0,
          "min_child_weight": 4,
          "silent": False,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "n_estimators": 200,
          "nthread": 16,
          "seed": 27}

# General steps for parameter tunning
# Step 1: Choose a high learning rate and low n_estimators. Then tune max_depth and min_child_weight
param_test = {
    'max_depth': range(3,10,2),
    'min_child_weight': range(1,6,2)
}

# Step 2: Tune gamma
param_test_1 = {
    'gamma': [0.1, 0.5, 1, 2, 5, 10]
}

# Step 3: Tune regularization parameters
param_test_2 = {
    'alpha': [i/10.0 for i in rage(10)],
    'beta': [i/10.0 for i in rage(10)]
}

# Step 4: Reducing Learning Rate
param_last_round = {"objective": custobj,
          "max_depth": 6,
          "learning_rate": 0.1,
          "gamma": 0,
          "min_child_weight": 4,
          "silent": False,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "n_estimators": 200,
          "nthread": 16,
          "seed": 27}

def scorer(estimator, X, y):
    ypred = estimator.predict(X)
    return 1.0 - np.average(np.abs([1-yh/y for (yh, y) in zip(ypred, y) if y != 0]))

gsearch = GridSearchCV(estimator = xgb.XGBRegressor(**params),
                        param_grid = param_test,
                        scoring=scorer, n_jobs=4,
                        iid=False, cv=5)
gsearch.fit(train_features, train_labels)
print gsearch.grid_scores_
print gsearch.best_params_
print gsearch.best_score_
