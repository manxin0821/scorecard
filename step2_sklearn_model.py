
import pickle
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
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
#df_german=pd.read_excel("german_credit.xlsx")
#df_german=pd.read_excel("df_after_vif.xlsx")
y=df_german["target"]
#x=df_german.ix[:,"Account Balance":"Foreign Worker"]
x=df_german.loc[:,"Account Balance":"Foreign Worker"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#solver='liblinear'
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


print("accuracy on the training subset:{:.3f}".format(classifier.score(X_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(classifier.score(X_test,y_test)))


'''
P0 = 50
PDO = 10
theta0 = 1.0/20
B = PDO/np.log(2)
A = P0 + B*np.log(theta0)
'''
def Score(probability):
    
    score = A-B*np.log(probability/(1-probability))
    return score

def List_score(pos_probablity_list):
    list_score=[]
    for probability in pos_probablity_list:
        score=Score(probability)
        list_score.append(score)
    return list_score

P0 = 50
PDO = 10
theta0 = 1.0/20
B = PDO/np.log(2)
A = P0 + B*np.log(theta0)
print("A:",A)
print("B:",B)
#print("test ok")

list_coef = list(classifier.coef_[0])
intercept= classifier.intercept_


probablity_list=classifier.predict_proba(x)

pos_probablity_list=[i[1] for i in probablity_list]

list_score=List_score(pos_probablity_list)
list_predict=classifier.predict(x)
df_result=pd.DataFrame({"label":y,"predict":list_predict,"pos_probablity":pos_probablity_list,"score":list_score})

df_result.to_excel("score_proba.xlsx")



list_vNames=df_german.columns

list_vNames=list(list_vNames[2:])


df_coef=pd.DataFrame({"variable_names":list_vNames,"coef":list_coef})


df_coef.to_excel("coef.xlsx")



save_classifier = open("scoreCard.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
#print("test over")