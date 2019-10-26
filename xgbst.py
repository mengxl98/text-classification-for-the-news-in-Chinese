#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from Tools import readbunchobj
import csv
import numpy as np

trainpath = "trainfile/tfdifspace_min3.dat"
train_set = readbunchobj(trainpath)

testpath = "trainfile/testspace_min3.dat"
test_set = readbunchobj(testpath)
lb = np.array(train_set.label)
x = [ int(x) for x in lb]

dtrain=xgb.DMatrix(train_set.tdm,label=x)
dtest=xgb.DMatrix(test_set.tdm)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':5,
    'lambda':4,
    'subsample':0.75,
    'colsample_bytree':0.8,
    'min_child_weight':6,
    'eta': 0.1,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)

ypred=bst.predict(dtest)

with open("resultxgb5.csv","w") as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(["id", "pred"])
     for file_name, expct_cate in zip(test_set.filenames, ypred):
         writer.writerow([file_name,expct_cate])


print("预测完毕!!!")
