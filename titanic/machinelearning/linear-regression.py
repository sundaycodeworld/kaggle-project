# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 09:24:27 2018

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


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'NameLength', 'Title']
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test  in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic['Survived'].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
accuracy = sum(predictions == titanic['Survived'])/len(predictions)

#accuracy = 0.7833894500561167
