# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import math

train = pd.read_csv('../datasets/train.csv', index_col='PassengerId')
test = pd.read_csv('../datasets/test.csv', index_col='PassengerId')

def deal_data(df):
    df['Name_length'] = df['Name'].apply(len)
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 1 if type(x) == str else 0)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['Age'][np.isnan(df['Age'])] = age_null_random_list
    df['Age'] = df['Age'].astype(int)

    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping Embarked
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    # Mapping Age
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4
    drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
    df = df.drop(drop_elements, axis=1)
    return df


def plot_data(df):
    colormap = plt.cm.RdBu
    corr_mat = df.astype(float).corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_mat, linewidths=0.1, vmax=1, square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()

kf = KFold(n_splits=5, random_state=0)

class SkearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def feature_importances(self):
        print(self.clf.feature_importances_)

if __name__ == '__main__':
    test = deal_data(test)
    train = deal_data(train)
    print("train shape: {}\t test shape:{}".format(train.shape, test.shape))
    plot_data(train)
    print("execute over")
