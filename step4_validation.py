
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
#混淆矩阵计算
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

#precision,TP/(TP+FP) 
precision=precision_score(y_true, y_pred)
print('model precision is:',precision)

#recall（sensitive）
sensitivity=recall_score(y_true, y_pred)
print('model sensitivity is:',sensitivity)
 
#F1 
f1Score=f1_score(y_true, y_pred)
print("f1_score:",f1Score)



probablity_list=classifier.predict_proba(X_test)

pos_probablity_list=[i[1] for i in probablity_list]


def AUC(y_true, y_scores):
    auc_value=0
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auc_value= auc(fpr,tpr) 
    #print("fpr:",fpr)
    #print("tpr:",tpr)
    #print("thresholds:",thresholds)
    if auc_value<0.5:
        auc_value=1-auc_value
    return auc_value

def Draw_roc(auc_value):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pos_probablity_list, pos_label=1)
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diagonal line') 
    plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % auc_value) 
    plt.title('ROC curve')  
    plt.legend(loc="lower right")

def AUC_performance(AUC):
    if AUC >=0.7:
        print("good classifier")
    if 0.7>AUC>0.6:
        print("not very good classifier")
    if 0.6>=AUC>0.5:
        print("useless classifier")
    if 0.5>=AUC:
        print("bad classifier,with sorting problems")
        
#Gini
def Gini(auc):
    gini=2*auc-1
    return gini

### ks
def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    
    total = df.groupby([score])[target].count()
    '''
    score
    0.00001    4
    0.00005    7
    0.00006    4
    0.00007    1
    0.00008    1
    '''
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all.index = range(len(all))
    all = all.sort_values(by=score,ascending=True)  
    #bad
    num_bad=all['bad'].sum()
    #good
    num_good= all['good'].sum()
    #cumulated bad
    all['badCumRate'] = all['bad'].cumsum() / num_bad
    #cumulated good
    all['goodCumRate'] = all['good'].cumsum() /num_good
    #bad-good
    ks_array = all.apply(lambda x: abs(x.badCumRate - x.goodCumRate), axis=1)
    #max(bad-good)
    ks=max(ks_array)
    return ks
        
#Auc验证，数据采用测试集数据
auc_value=AUC(y_test, pos_probablity_list)
print("AUC:",auc_value)

AUC_performance(auc_value)

Draw_roc(auc_value)


df=pd.DataFrame({'score':pos_probablity_list, 'target':y_test})

gini=Gini(auc_value)
print ("gini",gini)  
 

ks = KS(df,'score','target')
print("ks value:%.4f"%ks)

classifier_f.close()