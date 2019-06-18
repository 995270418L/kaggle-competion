import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.externals import joblib

def split_data(data):
    y = data['SalePrice']
    X = data.drop(['SalePrice'], axis=1)
    X = pd.get_dummies(X)
    print("X shape:{}, y shape{}".format(X.shape, y.shape))
    return X, y

def get_data():
    train_data = pd.read_csv("../datasets/train.csv")
    print(train_data.shape)
    return train_data

def get_test_data():
    data = pd.read_csv("../datasets/test.csv")
    print("source data shape: {}".format(data.shape))
    data = pd.get_dummies(data)
    print("test data shape: {}".format(data.shape))
    return data

def visualization(data):
    plt.plot(data['YrSold'], data['SalePrice'])
    plt.show()

def train_model(X, y):
    param_gv = {
        'subsample': [1]
    }
    bst = xgb.XGBRegressor(
        max_depth=22,
        min_child_weight=3,
        eta=0.5,
        n_estimators=1000,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        objective='reg:squarederror',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        seed=100
    )
    model = GridSearchCV(estimator=bst, param_grid=param_gv, scoring='r2', cv=10)
    model.fit(X, y)
    print("regression result:\n score: {}\t, best_params:{} \t ,best_score:{}".format(model.scorer_, model.best_params_, model.best_score_))
    return model

def score_model(X, y):
    bst = xgb.XGBRegressor(
        max_depth=22,
        min_child_weight=3,
        eta=0.5,
        n_estimators=1000,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        objective='reg:squarederror',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        seed=100
    )
    bst.fit(X, y, eval_metric='rmse')
    joblib.dump(bst, "train_model.m")
    # train_pred = bst.predict(X)
    # print("r2_score : {}".format(r2_score(y, train_pred)))
    # return bst

def get_result(model, test_data):
    y_pred = model.predict(test_data)
    result = pd.DataFrame([test_data['Id'], y_pred], columns=['Id', 'SalePrice'])
    result.to_csv(path_or_buf='../datasets/submission.csv')

if __name__ == '__main__':
    get_data()
    get_test_data()
    # train_data = get_data()
    # X, y = split_data(train_data)
    # model = score_model(X, y)
    # get_result(joblib.load("train_model.m"), get_test_data())
    # visualization(train_data)