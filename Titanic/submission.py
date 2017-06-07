# -*- coding : utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold

#####读取测试集#########
test =  pd.read_csv("F:/github/kaggle/Titanic/test.csv")
train = pd.read_csv("F:/github/kaggle/Titanic/train.csv")
test
##########对测试集进行清洗##########
test.describe()
##########缺失数据的处理###########
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Fare'] = train['Fare'].fillna(train['Fare'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
##########非数字数据的处理##########
train.loc[train['Sex'] =='male','Sex'] = 0
train.loc[train['Sex'] =='female','Sex'] = 1

test.loc[test['Sex'] =='male','Sex'] = 0
test.loc[test['Sex'] =='female','Sex'] = 1

lr = LogisticRegression()
selected_feature = ['Pclass','Age','SibSp', 'Parch', 'Fare']
lr.fit(train[selected_feature], train['Survived'])
predictions = lr.predict(test[selected_feature])
predictions
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("F:/github/kaggle/Titanic/submission.csv", index=False)
