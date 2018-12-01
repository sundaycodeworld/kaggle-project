# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 09:34:55 2018

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

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
accuracy = scores.mean()

#accuracy = 0.7878787878787877