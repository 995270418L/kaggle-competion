# -*- coding: UTF-8 –*-
'''
kaggle 大神代码  https://www.kaggle.com/abhinand05/predicting-housingprices-simple-approach-lb-top3
'''

import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
from IPython.display import display, HTML
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import missingno as msno  #search for missing value

# Import Sci-Kit Learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold

# Ensemble Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Package for stacking models
from vecstack import stacking
from joblib import dump, load

display(HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
"""))

def show_all(df):
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
        display(df)

def import_data():
    train_data = pd.read_csv("../datasets/train.csv", keep_default_na=False, index_col='Id')
    test_data = pd.read_csv("../datasets/test.csv", keep_default_na=False, index_col='Id')
    # train_data = pd.concat([train_data, test_data], axis=0, sort=True)
    # train_data.info()  # 这个命令可以看出哪些列的数据不全。
    # print(train_data.describe())
    return train_data, test_data

'''
 intuitively understand what's missing in our data and where it is missing.
'''
def plot_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    # plot missing value
    missing.plot.bar(figsize=(12, 8))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    plt.show()
    msno.matrix(df=df, figsize=(16, 8), color=(0, 0.2, 1), labels=True)
    plt.show()

'''
为数值列填充列的media值
为分类列填充最常见的分类
'''
def fill_missing_data(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]  # 得到有缺失值的列
    for column in list(missing.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)  # 分类列中最常见的分类
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)

'''
文本类型的数据换成数字（离散）
'''
def impute_cats(df):
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    object_cols_ind = []
    for i in object_cols:
        object_cols_ind.append(df.columns.get_loc(i))
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        # print("i:{}\t name:{}".format(i, object_cols[i]))
        df.iloc[:, i] = label_enc.fit_transform(df.iloc[:, i])

def corr(df):
    corr_mat = df[["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "BldgType",
                      "OverallQual", "OverallCond", "YearBuilt", "BedroomAbvGr", "PoolArea", "GarageArea",
                      "SaleType", "MoSold"]].corr()
    # f, ax = plt.subplots(figsize=(20, 20))
    # sns.heatmap(corr_mat, vmax=1, square=True)
    # plt.show() # 发现 OverallQual YearBuilt GarageArea 和 出售价格相关性较大
    # sns.lineplot(x='YearBuilt', y='SalePrice', data=df)
    # plt.show()
    # sns.lineplot(x='OverallQual', y='SalePrice', data=df)
    # plt.show()
    # sns.lineplot(x='GarageArea', y='SalePrice', data=df)
    # plt.show()
    # sns.distplot(df['SalePrice'])  # 销售价格的分布
    # plt.show()
    sns.catplot(x='SaleType', y='SalePrice', data=df, kind='bar', palette='muted')
    plt.show()

def r_f_model(X, y, kf=KFold(n_splits=5)):
    random_forest = RandomForestRegressor(n_estimators=1200, max_depth=15, min_samples_leaf=5, max_features=None,
                                          random_state=10, oob_score=True)
    y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=5, error_score='raise')
    print('random forest mean: {}'.format(y_pred.mean()))
    random_forest.fit(X, y)
    dump(random_forest, 'rf_model.joblib')

def xgboost_model(X, y, kf=KFold(n_splits=5)):
    xg_boost = XGBRegressor(learning_rate=0.01, n_estimators=6000, max_depth=4, min_child_weight=1, gamma=0,
                            subsample=1, colsample_bytree=0.2, objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=20, reg_alpha=0.00006)
    y_pred = cross_val_score(xg_boost, X, y, cv=kf, n_jobs=-1)
    print('xgboost mean:{}'.format(y_pred.mean()))
    xg_boost.fit(X, y, eval_metric='rmse')
    dump(xg_boost, 'xgb_model.joblib')

def gbm_model(X, y, kf=KFold(n_splits=5)):
    g_boost = GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01, max_depth=5, max_features='sqrt',
                                        min_samples_leaf=15, min_samples_split=10, loss='ls', random_state=30)
    y_pred = cross_val_score(g_boost, X, y, cv=kf, n_jobs=-1)
    print('gbm mean:{}'.format(y_pred.mean()))
    g_boost.fit(X, y)
    dump(g_boost, 'gmb_model.joblib')

def lgbm_model(X, y, kf=KFold(n_splits=5)):
    lgbm_boost = LGBMRegressor(objective='regression', num_leaves=6, learning_rate=0.01, n_estimators=6400, verbose=-1,
                             bagging_fraction=0.80, bagging_freq=4, bagging_seed=6,
                             feature_fraction=0.2, feature_fraction_seed=7)
    y_pred = cross_val_score(lgbm_boost, X, y, cv=kf, n_jobs=-1)
    print('lgbm mean:{}'.format(y_pred.mean()))
    lgbm_boost.fit(X, y, eval_metric='rmse')
    dump(lgbm_boost, 'lgmb_model.joblib')

def train_four_model(train_data):
    y = np.ravel(np.array(train_data[['SalePrice']]))
    X = train_data.drop('SalePrice', axis=1)
    r_f_model(X, y)
    xgboost_model(X, y)
    gbm_model(X, y)
    lgbm_model(X, y)

def integrated_models(train_data, test_data):
    print("train data shape:{}, test data shape:{}".format(train_data.shape, test_data.shape))

    y = np.ravel(np.array(train_data[['SalePrice']]))
    X = train_data.drop('SalePrice', axis=1)

    # load four models
    random_forest = load('rf_model.joblib')
    lightgbm = load('lgmb_model.joblib')
    g_boost = load('gmb_model.joblib')
    xg_boost = load('xgb_model.joblib')

    # model stacking
    models = [g_boost, xg_boost, lightgbm, random_forest]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    S_train, S_test = stacking(models, X_train, y_train, X_test, regression=True, mode='oof_pred_bag', metric=rmse, n_folds=5, random_state=25, verbose=2)

    print("S_train shape:{} \t S_test shape:{}".format(S_train.shape, S_test.shape))

    # 初始化第二层模型
    xgb_lev2 = XGBRegressor(learning_rate=0.1, n_estimators=500, max_depth=3, n_jobs=-1, random_state=17)
    # Fit the 2nd level model on the output of level 1
    xgb_lev2.fit(S_train, y_train)
    stacked_pred = xgb_lev2.predict(S_test)
    print("RMSE of Stacked Model: {}".format(rmse(y_test, stacked_pred)))

    y1_pred_L1 = models[0].predict(test_data)
    y2_pred_L1 = models[1].predict(test_data)
    y3_pred_L1 = models[2].predict(test_data)
    y4_pred_L1 = models[3].predict(test_data)
    S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1, y4_pred_L1]
    print("S_test_L1 shape: {}".format(S_test_L1.shape))
    test_stacked_pred = xgb_lev2.predict(S_test_L1)
    submission = pd.DataFrame()

    submission['Id'] = np.array(test_data.index)
    submission['SalePrice'] = test_stacked_pred
    submission.to_csv("submission.csv", index=False)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))

if __name__ == '__main__':
    train_data, test_data = import_data()
    fill_missing_data(train_data)
    print(train_data.isnull().sum().max())
    fill_missing_data(test_data)
    print(test_data.isnull().sum().max())
    impute_cats(train_data)
    impute_cats(test_data)
    # train_four_model(train_data)
    integrated_models(train_data, test_data)
    # integrated_models(train_data, test_data)
    # corr(train_data)

    # print("train data shape:{}\t test data shape: {}".format(train_data.shape, test_data.shape))
    # plot_missing(test_data)
    # fill_missing_data(train_data)
    # print(train_data.isnull().sum().max())