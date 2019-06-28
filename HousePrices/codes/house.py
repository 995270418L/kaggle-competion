from datetime import datetime

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
    return train_data, test_data

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

'''
TOP 1% 的代码也是混合别人的结果提升的，也挺垃圾的
'''
def explore_data_analysis(df=pd.DataFrame()):
    qualitative = df.select_dtypes(include='object')
    quantitative = df.select_dtypes(exclude='object')
    # normality_test using shapiro_wilk test
    normality_test = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01  # p_value 小于 0.01. 认为假设不成立，即不属于正态分布
    normal = quantitative.apply(normality_test)
    # print(normal.any()) # True 表示所有特征都不服从正态分布

    # 相关性 检测
    features = spearn_corrlation(list(quantitative.columns), list(qualitative.columns), train_data)
    # 降维数据并可视化， 查看聚类
    # dimension_reduce_visualization(features, train_data)

def dimension_reduce_visualization(features=[], train=pd.DataFrame()):
    # 使用的是 tsne 方法（数据量小的时候用）
    model = TSNE(n_components=2, perplexity=50, random_state=0)
    X = train[features].fillna(0).values
    tsne = model.fit_transform(X)
    print("tsne shape: {}".format(tsne.shape))
    std = StandardScaler()
    s = std.fit_transform(X)
    print("s shape:{}".format(s.shape))
    pca = PCA(n_components=30)
    pc = pca.fit_transform(s)   # pca 训练的数据都需要标准化
    print("pc shape:{}".format(pc.shape))
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pc)
    fr = pd.DataFrame({"tsne1": tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
    sns.lmplot(data = fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
    plt.show()
    print(np.sum(pca.explained_variance_ratio_))

def spearn_corrlation(quantitative=[], qualitative=[], train_data=pd.DataFrame()):
    qual_encoded = []
    for q in qualitative:
        encode(q, train_data)
        qual_encoded.append(q + "_E")
    features = quantitative + qual_encoded
    return features

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

def feature_selection(train=pd.DataFrame(), test=pd.DataFrame()):
    explore_data_analysis(train)
    train = train[train['GrLivArea'] < 4500]
    train['SalePrice'] = train['SalePrice'].map(lambda x : math.log(1+x))
    y = train['SalePrice']
    train_features = train.drop("SalePrice", axis=1)
    test_features = test
    features = pd.concat([train_features, test_features], sort=False)
    features['MSSubClass'] = features['MSSubClass'].apply(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features["PoolQC"] = features["PoolQC"].fillna("None")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))  # 填充出现频率最高的数(众数)
    object = list(features.select_dtypes(include='object').columns)
    features.update(features[object].fillna('None'))
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median())) # 填充中位数
    numeric_dtypes = list(features.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns)
    features.update(features[numeric_dtypes].fillna(0))

    # 统计所有偏度 > 0.5 的features
    skew_features = features[numeric_dtypes].apply(lambda x: skew(x)).sort_values(ascending=False) # 默认是从小到大排序， 这里不进行排序，节约时间

    high_skew = skew_features[skew_features > 0.5]
    skew_high_index = high_skew.index
    for i in skew_high_index:
        # 对偏度大于0.5的列进行box-cox 正态化矫正
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
    features = features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
    features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']
    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

    features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                     features['1stFlrSF'] + features['2ndFlrSF'])

    features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                                   features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

    features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                                  features['EnclosedPorch'] + features['ScreenPorch'] +
                                  features['WoodDeckSF'])
    features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    # 对所有变量进行扩充处理
    final_features = pd.get_dummies(features)
    # 分离X_train, Y_train, X_test
    X = final_features.iloc[:len(y), :]
    test = final_features.iloc[len(y):, :]
    # print("X shape:{}\t y shape:{}\t test shape:{}".format(X.shape, y.shape, test.shape))
    overfit = []
    for i in X.columns:
        counts = X[i].value_counts()
        zeros = counts.iloc[0]  # 获取 i 列中的众数
        if zeros / len(X[i]) * 100 > 99.94:
            overfit.append(i) # 占比超过 99.94 数据容易过拟合
    X = X.drop(overfit, axis=1)
    test = test.drop(overfit, axis=1)
    print("X shape:{}\t y shape:{}\t test shape:{}".format(X.shape, y.shape, test.shape))
    return X, y, test

def rmlse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X, y, kfolds):
    return np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds, n_jobs=-1))

def model_selection(X, y, test):
    kfolds = KFold(n_splits=10, random_state=42)
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5, 100]
    alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    X = RobustScaler().fit_transform(X)
    ridge = RidgeCV(alphas_alt, cv=kfolds)
    # lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
    # elastic_net = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
    # svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003, ))
    # gbr = GradientBoostingRegressor(n_estimators=5000, max_depth=4, learning_rate=0.05, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
    # lightgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75,
    #                          bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1, n_jobs=-1)
    # xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7,colsample_bytree=0.7,
    #                        objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006, n_jobs=-1)
    # stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elastic_net, gbr, xgboost, lightgbm), meta_regressor=xgboost, use_features_in_secondary=True, n_jobs=-1)

    score = cv_rmse(ridge, X, y, kfolds)
    print("RIDGE: {:.4f} ({:.4f})\n".format( score.mean(), score.std()), datetime.now())
    ridge.fit(X, y)
    print("Bast alphas_alt: {}".format(ridge.alpha_))
    # score = cv_rmse(lasso, X, y, kfolds)
    # print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())
    #
    # score = cv_rmse(elastic_net, X, y, kfolds)
    # print("ELASTIC_NET: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())
    # score = cv_rmse(svr, X, y, kfolds)
    # print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())
    # score = cv_rmse(lightgbm, X, y, kfolds)
    # print("LIGHTGBM: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())
    # score = cv_rmse(gbr, X, y, kfolds)
    # print("GBR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())
    # score = cv_rmse(xgboost, X, y, kfolds)
    # print("XGBOOST: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())
    #
    # print("######### START FIT #############")
    # print("stack_gen")
    # stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
    #
    # print("elasticnet")
    # elastic_net_model = elastic_net.fit(X, y)
    #
    # print("lasso")
    # lasso_model = lasso.fit(X, y)
    #
    # print("ridge")
    # ridge_model = ridge.fit(X, y)
    #
    # print("Svr")
    # svr_model = svr.fit(X, y)
    #
    # print('GradientBoosting')
    # gbr_model_full_data = gbr.fit(X, y)
    #
    # print('xgboost')
    # xgb_model_full_data = xgboost.fit(X, y)
    #
    # print('lightgbm')
    # lgb_model_full_data = lightgbm.fit(X, y)
    #
    # def blend_model_predict(X_tmp):
    #     return 0.1 * elastic_net_model.predict(X_tmp) + 0.05 * lasso_model.predict(X_tmp) + 0.1 * ridge_model.predict(X_tmp) + \
    #            0.1 * svr_model.predict(X_tmp) + 0.1 * gbr_model_full_data.predict(X_tmp) + 0.15 * xgb_model_full_data.predict(X_tmp) + \
    #            0.1 * lgb_model_full_data.predict(X_tmp) + 0.3 * stack_gen_model.predict(np.array(X_tmp))
    # print("RMSLE score on train data: {}".format(rmlse(y, blend_model_predict(X))))
    # print("Predict submission")
    # submission = pd.read_csv("../datasets/sample_submission.csv")
    # print("test data head:{}".format(test.head()))
    # submission.loc[:, 1] = blend_model_predict(test)
    # submission.to_csv("submission.csv")

def blending_result():
    train_submission = pd.read_csv("submission.csv", index_col='Id')
    other_best_submission = pd.read_csv("../datasets/best_submission.csv", index_col='Id')
    result = np.floor(train_submission * 0.3 + 0.7 * other_best_submission)
    result.to_csv("result_submission.csv")


if __name__ == '__main__':
    train_data, test_data = get_data()
    X, y, test = feature_selection(train_data, test_data)
    # print('X head: {}, \n########################\ny head: {} \n########################\n test head:{}'.format(X.head, y.head, test.head))
    model_selection(X, y, test)
    # blending_result()
