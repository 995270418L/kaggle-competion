import math
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV  # ElasticNet regression 其中的 penalty 函数是 L1 + L2 ，也就是 Lasso 和Ridge 的 penalty 之和
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler  # 数据预处理的标准化，(X - mean) / std 数据聚集在0 附近， 均值为1
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR  # SVM 的重要分支，回归算法的一种
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from xgboost import XGBRegressor, XGBRFRegressor

from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor  # 集成学习

from scipy import stats
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p # box-cox 转换, 将数据尽量转换为正态分布的数据
from joblib import dump, load
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
    R_X = RobustScaler().fit_transform(X)
    # ridge 训练
    ridge = RidgeCV(cv=kfolds)
    ridge.fit(R_X, y)
    # dump(ridge, '../models/ridgecv.joblib')
    print("best ridge param :{}".format(ridge.alpha_))
    ridge_no_cv = Ridge(alpha=ridge.alpha_)
    ridge_no_cv.fit(R_X, y)
    dump(ridge_no_cv, '../models/ridge.joblib')
    # print("X head:{} \n################\n y head :{}".format(X.head(), y.head()))
    # lasso 训练
    # lassocv = LassoCV(cv=kfolds, n_jobs=-1, random_state=50)
    # lassocv.fit(R_X, y)
    # dump(lassocv, '../models/lassocv.joblib')
    # print(lassocv.alpha_)

    # Elastic Net CV
    # elastic_net_cv = ElasticNetCV(cv=kfolds, n_jobs=-1, random_state=50)
    # elastic_net_cv.fit(R_X, y)
    # dump(elastic_net_cv, '../models/elastic_net_cv.joblib')
    # print(elastic_net_cv.alpha_)

    '''
    ensemble 算法调优，使用 GridSearch 
    '''
    # 随机森林
    params = {
        "n_estimators": [900],
        "max_depth": [2]
    }
    # rfr = XGBRFRegressor(
    #     n_estimators=100,
    #     random_state=50,
    #     n_jobs=-1,
    #     objective='reg:squarederror',
    #     learning_rate=1
    # )
    # xgboost = XGBRegressor(
    #     learning_rate=0.01,
    #     objective='reg:squarederror',
    #     n_jobs=-1,
    #     random_state=50,
    #     n_estimators=3000,
    #     max_depth=4
    # )
    # xgboost.fit(X, y)
    # dump(xgboost, '../models/xgboost.joblib')
    # lgdm = LGBMRegressor(
    #     n_estimators=500,
    #     max_depth=4,
    #     learning_rate=0.01,
    #     num_leaves=14,
    #     objective='regression'
    # )
    # lgdm.fit(X, y)
    # gbr = GradientBoostingRegressor(
    #     max_depth=2,
    #     n_estimators=900,
    #     learning_rate=0.1,
    #     random_state=50,
    #     loss='huber',  # 数据噪音多使用huber，少就使用ls
    #     alpha=0.1,
    #     max_features='auto'
    # )
    # gsc = GridSearchCV(param_grid=params, estimator=gbr, scoring='neg_mean_squared_error', cv=kfolds, n_jobs=5)
    # gsc.fit(X, y)
    # print("rfr model score :{}, \nbest params: {}\n".format(np.sqrt(-gsc.best_score_), gsc.best_params_))\


def stack_model(X, y, test):
    elastic = load("../models/elastic_net_cv.joblib")
    gbr = load("../models/gbr.joblib")
    lasso = load("../models/lassocv.joblib")
    lgdm = load("../models/lgdm.joblib")
    ridge = load("../models/ridge.joblib")
    xgboost = load("../models/xgboost.joblib")
    stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elastic, gbr, xgboost, lgdm),
                                    meta_regressor=xgboost, use_features_in_secondary=True, n_jobs=-1)
    stack_gen.fit(np.array(X), np.array(y))
    dump(stack_gen, '../models/result.joblib')

def canculate_result(test):
    stack_gen = load('../models/result.joblib')
    data = stack_gen.predict(np.array(test))
    sample_submission = pd.read_csv('../datasets/sample_submission.csv', index_col='Id')
    data = np.expm1(data)
    print("data shape :{}".format(data.shape))
    sample_submission.iloc[:, 0] = data
    sample_submission.to_csv('../datasets/result.csv')

def rmlse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X, y, cv):
    return np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))

def explor_numeric_data(df=pd.DataFrame()):
    pass

if __name__ == '__main__':
    train_data, test_data = get_data()
    train_len = len(train_data)
    data = pd.concat([train_data, test_data], sort=False)
    salePrice = data.loc[:, 'SalePrice']
    data = data.drop('SalePrice', axis=1)
    fill_null_data(data)
    new_data = pd.get_dummies(data)
    X = new_data.iloc[:train_len, :]
    y = salePrice.iloc[:train_len]
    y = y.apply(lambda x: np.log1p(x))
    test = new_data.iloc[train_len:, :]
    print('X shape {}, y shape {} test shape{}'.format(X.shape, y.shape, test.shape))
    # print('X head: {}, \n########################\ny head: {} \n########################\n test head:{}'.format(X.head, y.head, test.head))
    # model_selection(X, y, test)
    # stack_model(X, y, test)
    canculate_result(test)
