

from sklearn.model_selection import KFold
import numpy as np
X = np.arange(48).reshape(12,4)

kf = KFold(n_splits=5,shuffle=False)
for train_index , test_index in kf.split(X):
    print('train_index:%s , test_index: %s ' %(train_index,test_index))
    
# cross validation
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df_german=pd.read_excel("german_woe.xlsx")
y=df_german["target"]
x=df_german.ix[:,"Account Balance":"Foreign Worker"]

model=LogisticRegression()
list_scores=cross_val_score(model,x,y,cv=5,scoring='accuracy')
average_score=list_scores.mean()
print("cross validation mean score:",average_score)




from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
model2=LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model2.fit(X_train,y_train)
y_true=y_test
y_pred=model2.predict(X_test)
accuracyScore = accuracy_score(y_true, y_pred)
print('model accuracy is:',accuracyScore)