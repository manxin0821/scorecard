
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

df_german=pd.read_excel("german_woe.xlsx")
y=df_german["target"]
x=df_german.ix[:,"Account Balance":"Foreign Worker"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#pvs = []  
logit_mod = sm.Logit(y_train, X_train)
logitres = logit_mod.fit(disp=False)
predict_y=logitres.predict(x)


summary_logistic=logitres.summary()
#pvs.append([item, logitres.pvalues[item]])
file_logistic_summary=open("summary_logistic.txt",'w')
file_logistic_summary.write(str(summary_logistic))
file_logistic_summary.close()



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
    fpr, tpr, thresholds = metrics.roc_curve(y, predict_y, pos_label=1)
  
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
        

auc_value=AUC(y, predict_y)
print("AUC:",auc_value)

AUC_performance(auc_value)

Draw_roc(auc_value)




