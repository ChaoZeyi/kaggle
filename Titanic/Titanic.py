# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
train = pd.read_csv("F:/github/kaggle/Titanic/train.csv")
train
train.describe()
train['Age'].median()
###########数据清洗：把nan数据补全，把非数值型数据转换成数值型数据###################
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna('S')

train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

lr = LinearRegression()
kf = KFold(train.shape[0], n_folds=3)
train['Survived'].count()
predictions = np.array([])
for train_temp, test_temp in kf:
    train_data = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]# 选择要使用的特征
    train_data = train_data.iloc[train_temp, :]
    train_target = train['Survived']
    train_target = train_target.iloc[train_temp]
    lr.fit(train_data, train_target)
    test_data = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].iloc[test_temp, :]
    test_target = lr.predict(test_data)
    predictions=np.append(predictions, test_target)
predictions
#predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
predictions

sum(predictions[predictions == train['Survived']])
accuracy = sum(predictions[predictions == train['Survived']])/train.shape[0]
accuracy
test =  pd.read_csv("F:/github/kaggle/Titanic/test.csv")
test
##########对测试集进行清洗##########
test.describe()
##########缺失数据的处理###########
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
##########非数字数据的处理##########
test.loc[test['Sex'] =='male','Sex'] = 0
test.loc[test['Sex'] =='female','Sex'] = 1
predictions = lr.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']])
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
submission = pd.DataFrame({"PassengerId":test['PassengerId'], "Survived": predictions})
submission.to_csv("F:/github/kaggle/Titanic/kaggle.csv", index=False)
