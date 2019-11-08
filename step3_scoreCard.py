

import copy
import numpy as np
import pandas as pd
import step1_woe
import step2_sklearn_model

df=step1_woe.df
dict_woe=step1_woe.scores
dict_woe1=copy.deepcopy(dict_woe)

variable_names=list(df.columns)[:-1]
coef=pd.read_excel("coef.xlsx")

B=step2_sklearn_model.B

df_total=pd.DataFrame()

for i in range(len(variable_names)):
    
    varName=variable_names[i]
    list_subVar_woe=dict_woe1[varName]
    
    numbers=len(list_subVar_woe)
    
    for number in range(numbers):
        var_Subdivision=list_subVar_woe[number]
        woe=var_Subdivision[1]
        coef1=float(coef[coef.variable_names==varName]['coef'])
        
        score=woe*coef1*B*(-1)
        var_Subdivision.append(score)
        var_Subdivision.append(varName)

    names=['span','woe','score','varName']
    df_subVar_woe=pd.DataFrame(data=list_subVar_woe,columns=names)
    df_total=pd.concat([df_total,df_subVar_woe],axis=0)
        
df_total.to_excel("score_card.xlsx")






