import math
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV  # ElasticNet regression 其中的 penalty 函数是 L1 + L2 ，也就是 Lasso 和Ridge 的 penalty 之和
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler  # 数据预处理的标准化，(X - mean) / std 数据聚集在0 附近， 均值为1
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR  # SVM 的重要分支，回归算法的一种
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor  # 集成学习

from scipy import stats
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p # box-cox 转换, 将数据尽量转换为正态分布的数据

import seaborn as sns
import numpy as np

def get_data():
    train_data = pd.read_csv("../datasets/train.csv", index_col='Id')
    test_data = pd.read_csv("../datasets/test.csv", index_col='Id')
    return train_data, test_data

def null_data_print(data=pd.DataFrame):
    '''
    MSZoning LotFrontage Exterior1st Exterior2nd BsmtFullBath BsmtHalfBath KitchenQual Functional GarageYrBlt GarageCars SaleType Electrical : 众数
    MasVnrArea BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF GarageArea ： 中位数
    Alley  Utilities MasVnrType BsmtQual BsmtCond BsmtFinType1 BsmtFinType2 FireplaceQu  GarageType GarageFinish GarageQual GarageCond PoolQC Fence MiscFeature ： 一种类别 "None"
    '''
    null_test_data = data.isnull().sum()
    null_data = null_test_data[null_test_data > 0]
    print(null_data)
    return list(null_data.index)

def fill_null_data(data=pd.DataFrame):
    mode_data = ["MSZoning", "LotFrontage", "Exterior1st", "Exterior2nd", "BsmtFullBath", "BsmtHalfBath", "KitchenQual", "Functional", "GarageYrBlt", "GarageCars", "SaleType", "Electrical"]
    for i in mode_data:
        data.update(data[i].fillna(data[i].mode()[0]))
    median_data = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageArea"]
    for i in median_data:
        data.update(data[i].fillna(data[i].median()))
    type_data = ["Alley", "Utilities", "MasVnrType", "BsmtExposure", "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
    data.update(data[type_data].fillna('None'))
    return data

def model_selection(X, y, test):
    kfolds = KFold(n_splits=10, random_state=42)
    model_test(X, y, kfolds)

def model_test(X, y, kfolds):
    X = RobustScaler().fit_transform(X)
    #ridge 训练
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    ridge = RidgeCV(alphas=alphas_alt, cv=kfolds, fit_intercept=True, normalize=False, scoring="neg_mean_squared_error", store_cv_values=False)
    ridge.fit(X, y)
    print("best ridge param :{}".format(ridge.alpha_))
    # lasso 训练
    alphas = [0.9, 1, 2, 3, 4, 5, 6, 10]
    lasso = LassoCV(alphas, cv=kfolds, n_jobs=-1, random_state=50, max_iter=10000)
    print(lasso.alphas_)


def explor_numeric_data(df=pd.DataFrame()):
    pass

if __name__ == '__main__':
    train_data, test_data = get_data()
    train_len = len(train_data)
    data = pd.concat([train_data, test_data], sort=False)
    salePrice = data.loc[:, 'SalePrice']
    data = data.drop('SalePrice', axis=1)
    fill_null_data(data)
    null_data_print(data)
    new_data = pd.get_dummies(data)
    X = new_data.iloc[:train_len, :]
    y = salePrice.iloc[:train_len]
    test = new_data.iloc[train_len:, :]
    print('X shape {}, y shape {} test shape{}'.format(X.shape, y.shape, test.shape))
    model_selection(X, y, test)
    # print('X null sum:{}\t test numm sum: {}'.format(X.isnull().sum(), test.isnull.sum()))
    # print("X: {}".format(X.isnull().sum()))
    # print("test: {}".format(test.isnull().sum()))
    # test_null_data = null_data_print(test_data)