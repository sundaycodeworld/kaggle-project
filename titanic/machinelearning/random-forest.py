# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 09:37:31 2018

@author: zh
"""
import pandas as pd
import numpy as np

titanic = pd.read_csv('train.csv')

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

titanic['Embarked'] = titanic['Embarked'].replace('nan', np.nan).fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))

import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''
titles = titanic['Name'].apply(get_title)
#pd.value_counts(titles)

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9, 'Capt': 10, 'Ms': 11, 'Jonkheer': 12, 'Don':13, 'Sir':14, 'Countess':15, 'Lady':16, 'Mme':17}
for k,v in title_mapping.items():
    titles[titles==k]=v
#pd.value_counts(titles)
titanic['Title'] = titles

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)
accuracy = scores.mean()
#accuracy = 0.7856341189674523

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)
accuracy = scores.mean()
#accuracy = 0.8159371492704826

alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=9, min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)
accuracy = scores.mean()
#accuracy = 0.8316498316498316
