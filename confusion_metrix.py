
import pickle
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import cross_val_score
import statsmodels.api as sm

from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

classifier_f = open("scoreCard.pickle", "rb")
classifier = pickle.load(classifier_f)

df_german=pd.read_excel("german_woe.xlsx")
y=df_german["target"]
x=df_german.ix[:,"Account Balance":"Foreign Worker"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

y_true=y_test
y_pred=classifier.predict(X_test)
accuracyScore = accuracy_score(y_true, y_pred)
print('model accuracy is:',accuracyScore)

#precision,TP/(TP+FP) （
precision=precision_score(y_true, y_pred)
print('model precision is:',precision)

#recall（sensitive）
sensitivity=recall_score(y_true, y_pred)
print('model sensitivity is:',sensitivity)
 
#F1 
f1Score=f1_score(y_true, y_pred)
print("f1_score:",f1Score)
