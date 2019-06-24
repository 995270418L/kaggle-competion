import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import Imputer
from scipy import stats
import seaborn as sns
import numpy as np

def split_data(data):
    y = data['SalePrice']
    X = data.drop(['SalePrice'], axis=1)
    X = pd.get_dummies(X)
    test_data = X.iloc[1460:]
    X = X.iloc[:1460]
    y = y.iloc[:1460]
    return X, y, test_data

def get_data():
    train_data = pd.read_csv("../datasets/train.csv", index_col='Id')
    test_data = pd.read_csv("../datasets/test.csv", index_col='Id')
    # train_data = pd.concat([train_data, test_data], axis=0, sort=True)
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
        'n_estimators': [1000, 500, 800]
    }
    bst = xgb.XGBRegressor(
        max_depth=22,
        min_child_weight=3,
        eta=0.5,
        # n_estimators=1000,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        eval_metric='rmse',
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

def all_data_model(X, y):
    X = nan_imputer(X)
    bst = xgb.XGBRegressor(
        max_depth=22,
        min_child_weight=3,
        eta=0.5,
        n_estimators=800,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        objective='reg:squarederror',
        eval_metric='rmse',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        seed=100
    )
    bst.fit(X, y)
    selection = SelectFromModel(bst, prefit=True)
    X_new = selection.transform(X)
    model = xgb.XGBRegressor(
        max_depth=15,
        min_child_weight=3,
        eta=0.5,
        n_estimators=110,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        eval_metric='rmse',
        objective='reg:squarederror',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.5,
        seed=100
    )
    model.fit(X_new, y)
    y_pred = model.predict(X_new)
    print("r2 score:{}".format(r2_score(y, y_pred)))
    print("X new:{}".format(X_new[:2]))
    print("X new:{}\t type x :{} ".format(X_new.shape, type(X_new)))

def selection_model_cv(X, y):
    bst = xgb.Booster(model_file="all_data_model.model")
    selection = SelectFromModel(bst, prefit=True)
    X_new = selection.transform(X)
    # cv train
    param_gv = {
        'n_estimators': [100, 200, 300],
    }
    bst = xgb.XGBRegressor(
        max_depth=15,
        min_child_weight=3,
        eta=0.5,
        # n_estimators=100,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        eval_metric='rmse',
        objective='reg:squarederror',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        seed=100
    )
    model = GridSearchCV(estimator=bst, param_grid=param_gv, scoring='r2', cv=10)
    model.fit(X_new, y)
    print("regression result:\n score: {}\t, best_params:{} \t ,best_score:{}".format(model.scorer_, model.best_params_,
                                                                                      model.best_score_))
def selection_model(X, y):
    X = nan_imputer(X)
    bst_old = xgb.Booster(model_file="all_data_model.model")
    selection = SelectFromModel(bst_old, prefit=True)
    X_new = selection.transform(X)
    model = xgb.XGBRegressor(
        max_depth=15,
        min_child_weight=3,
        eta=0.5,
        n_estimators=110,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        eval_metric='rmse',
        objective='reg:squarederror',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.5,
        seed=100
    )
    model.fit(X_new, y)
    y_pred = model.predict(X_new)
    print("r2 score:{}".format(r2_score(y, y_pred)))
    return X_new.columns

def score_model(X, y):
    X = nan_imputer(X)
    bst = xgb.XGBRegressor(
        max_depth=22,
        min_child_weight=3,
        eta=0.5,
        n_estimators=800,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        objective='reg:squarederror',
        eval_metric='rmse',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        seed=100
    )
    bst.fit(X, y)
    # selection model
    selection = SelectFromModel(bst, prefit=True)
    X_new = selection.transform(X)

    # do grid search
    param_gv = {
        'subsample': [0.95, 0.98, 1],
        # 'colsample_bylevel': [0.4, 0.5, 0.6]
        # 'n_estimators': [100, 150, 200]
    }
    bst = xgb.XGBRegressor(
        max_depth=15,
        min_child_weight=3,
        eta=0.5,
        n_estimators=110,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        eval_metric='rmse',
        objective='reg:squarederror',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.5,
        seed=100
    )
    model = GridSearchCV(estimator=bst, param_grid=param_gv, scoring='r2', cv=10)
    model.fit(X_new, y)
    print("regression result:\n score: {}\t, best_params:{} \t ,best_score:{}".format(model.scorer_, model.best_params_,
                                                                                      model.best_score_))
    # bst_new = xgb.XGBRegressor(
    #     max_depth=22,
    #     min_child_weight=3,
    #     eta=0.5,
    #     n_estimators=800,
    #     learning_rate=0.1,
    #     gamma=0,
    #     subsample=1,
    #     objective='reg:squarederror',
    #     eval_metric='rmse',
    #     nthread=12,
    #     colsample_bytree=0.75,
    #     colsample_bylevel=0.7,
    #     seed=100
    # )
    # X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=100)
    # bst_new.fit(X_train, y_train)
    # y_pred = bst_new.predict(X_test)
    # print("r2 score: {}".format(r2_score(y_test, y_pred)))

    # print("x new shape:{}".format(X_new.shape))
    # bst.save_model("train_model.model")
    # # plot feature importance
    # plot_importance(bst)
    # plt.show()
    # # features = X.columns
    # # plt.bar(bst.f, bst.feature_importances_)
    # # plt.show()
    # y_pred = bst.predict(X)
    # print("r2 score: {}".format(r2_score(y, y_pred)))
    # return bst

def get_result(model, test_data):
    y_pred = model.predict(test_data)
    result = pd.DataFrame([test_data['Id'], y_pred], columns=['Id', 'SalePrice'])
    result.to_csv(path_or_buf='../datasets/submission.csv')

def train_save_model():
    train_data = get_data()
    X, y, test_data = split_data(train_data)
    score_model(X, y)

def get_result():
    train_data = get_data()
    X, y, test_data = split_data(train_data)
    # bst = score_model(X, y)
    bst = xgb.Booster(model_file="train_model.model")
    test_xgb_data = xgb.DMatrix(test_data)
    test_pred = bst.predict(test_xgb_data)
    test_series_pred = pd.Series(test_pred, name='SalePrice')
    result = pd.DataFrame()
    result['Id'] = test_data['Id']
    result['SalePrice'] = test_series_pred
    print("result head:{}".format(result.head))
    result.to_csv(path_or_buf='submission.csv', index=False)

'''
垃圾
'''
def lasso_regression(X, y):
    X = nan_imputer(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = Lasso()
    model.fit(X_train, y_train)
    print('系数矩阵:{}\n'.format(model.coef_))
    print('线性回归模型:{}\n'.format(model))
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print("accuracy :{}".format(accuracy))

def train_test_split_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    X_train = nan_imputer(X_train)
    X_test = nan_imputer(X_test)
    bst = xgb.XGBRegressor(
        max_depth=22,
        min_child_weight=3,
        eta=0.5,
        n_estimators=100,
        learning_rate=0.1,
        gamma=0,
        subsample=1,
        objective='reg:squarederror',
        eval_metric='rmse',
        nthread=12,
        colsample_bytree=0.75,
        colsample_bylevel=0.7,
        seed=100
    )
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print("Accurancy: %.2f%%" % (accuracy * 100.0))
    thresholds = sorted(bst.feature_importances_)
    print(thresholds)
    for thresh in thresholds:
        selection = SelectFromModel(bst, threshold=thresh, prefit=True)
        select_x_train = selection.transform(X_train)
        selection_model = xgb.XGBRegressor(
            max_depth=22,
            min_child_weight=3,
            eta=0.5,
            n_estimators=100,
            learning_rate=0.1,
            gamma=0,
            subsample=1,
            objective='reg:squarederror',
            eval_metric='rmse',
            nthread=12,
            colsample_bytree=0.75,
            colsample_bylevel=0.7,
            seed=100
        )
        selection_model.fit(select_x_train, y_train)
        select_x_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_x_test)
        accuracy = r2_score(y_test, y_pred)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_x_train.shape[1], accuracy * 100.0))

def nan_imputer(data):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data)
    return imputer.transform(data)

def explore_data_analysis(df=pd.DataFrame()):
    qualitative = df.select_dtypes(include='object')
    quantitative = df.select_dtypes(exclude='object')
    # normality_test using shapiro_wilk test
    normality_test = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01  # p_value 小于 0.01. 认为假设不成立，即不属于正态分布
    normal = quantitative.apply(normality_test)
    # print(normal.any()) # True 表示所有特征都不服从正态分布
    # 相关性 检测
    qual_encoded = []
    for q in qualitative.columns:
        encode(q, train_data)
        qual_encoded.append(q + "_E")
    features = list(quantitative.columns) + qual_encoded
    # print("features :{}".format(features))
    # spearman(features, train_data)
    plt.figure(1)
    corr = train_data[list(quantitative.columns) + ['SalePrice']].corr()
    sns.heatmap(corr, vmax=.8)
    plt.figure(2)
    corr = train_data[qual_encoded + ['SalePrice']].corr()
    sns.heatmap(corr, vmax=.8)
    plt.figure(3)
    corr = pd.DataFrame(np.zeros([len(quantitative.columns) + 1, len(qual_encoded) + 1]), index=list(quantitative.columns) + ['SalePrice'], columns=qual_encoded + ['SalePrice'])
    for q1 in list(quantitative.columns) + ['SalePrice']:
        for q2 in qual_encoded + ['SalePrice']:
            corr.loc[q1, q2] = train_data[q1].corr(train_data[q2])
    sns.heatmap(corr)
    plt.show()

def encode(feature, df=pd.DataFrame()):
    ordering = pd.DataFrame()
    ordering['val'] = df[feature].unique()  # values unique
    ordering.index = ordering.val
    ordering['spmean'] = df[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']  # 按 feature 统计对应 SalePrice 的平均值
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['ordering'].to_dict()
    # 替换所有的 feature 的值
    for key, value in ordering.items():
        df.loc[df[feature] == key, feature + "_E"] = value

def spearman(features, df=pd.DataFrame()):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [df[f].corr(df['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    plt.show()

def feature_selection(df=pd.DataFrame()):
    corr_mat = df.corr()
    sns.heatmap(corr_mat, vmax=.8, square=True)
    corr_mat.nla
    print(corr_mat.inde)
    plt.show()
    #

if __name__ == '__main__':
    train_data = get_data()
    explore_data_analysis(train_data)
    # X, y, data = split_data(train_data)
    # all_data_model(X, y)
    # score_model(X, y)
    # train_test_split_model(X, y)
    # train_model(X, y)
    # train_save_model()
    # get_result()
    # visualization(train_data)