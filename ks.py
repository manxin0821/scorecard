
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split

classifier_f = open("scoreCard.pickle", "rb")
classifier = pickle.load(classifier_f)

df_german=pd.read_excel("german_woe.xlsx")
y=df_german["target"]
x=df_german.ix[:,"Account Balance":"Foreign Worker"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


probablity_list=classifier.predict_proba(X_test)

pos_probablity_list=[i[1] for i in probablity_list]

pos_probablity_list1=[round(i,5) for i in pos_probablity_list]

df=pd.DataFrame({'score':pos_probablity_list1, 'target':y_test})
'''
     score  target
993   0.38       1
859   0.45       1
298   0.21       0
553   0.07       0
672   0.34       0
971   0.54       1
27    0.10       0
231   0.19       0
306   0.31       0
706   0.20       0
496   0.06       0
558   0.13       0
'''

total = df.groupby(['score'])['target'].count()
'''
   score
0.01     1
0.02     3
0.03     5
0.04    13
0.05    11
0.06    13
0.07     7
    '''
bad = df.groupby(['score'])['target'].sum()
'''
score
0.01    0
0.02    0
0.03    0
0.04    1
0.05    0
0.06    0
0.07    1
'''
all = pd.DataFrame({'total':total, 'bad':bad})
all['good'] = all['total'] - all['bad']
all['score'] = all.index
all.index = range(len(all))
all = all.sort_values(by='score',ascending=True)  
# Bad rate
num_bad=all['bad'].sum()
# Good rate
num_good= all['good'].sum()
# Cumulated bad rate
all['badCumRate'] = all['bad'].cumsum() / num_bad
# Cumulated good rate
all['goodCumRate'] = all['good'].cumsum() /num_good
# bad-good
ks_array = all.apply(lambda x: abs(x.badCumRate - x.goodCumRate), axis=1)
all['ks_array ']=ks_array 
#max (bad-good)
ks=max(ks_array)


print("ks value:%.4f"%ks)
classifier_f.close()
