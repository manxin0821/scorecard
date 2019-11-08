# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

def woe_more(item, df, df_woe):
    xitem = np.array(df[item])
    y = df.loc[:, 'target']
    y = np.array(y)
    x = []
    for k in xitem:
        x.append([k])
    leastentro = 100
    tt_bad = sum(y)
    tt_good = len(y) - sum(y)
    l = []
    #3箱
    for m in range(10):
        y_pred = KMeans(n_clusters=4, random_state=m).fit_predict(x)
        a = [[[], []], [[], []], [[], []], [[], []]]  
        for i in range(len(y_pred)):
            a[y_pred[i]][0].append(x[i][0])
            a[y_pred[i]][1].append(y[i])
        a = sorted(a, key=lambda x: sum(x[0]) / len(x[0]))
        if sum(a[0][1]) / len(a[0][1]) >= sum(a[1][1]) / len(a[1][1]) >= sum(a[2][1]) / len(a[2][1]) >= sum(a[3][1]) \
                / len(a[3][1]) or sum(a[0][1]) / len(a[0][1]) <= sum(a[1][1]) / len(a[1][1]) \
                <= sum(a[2][1]) / len(a[2][1]) <= sum(a[3][1]) / len(a[3][1]):
            entro = 0
            for j in a:
                entro = entro + (- (len(j[1]) - sum(j[1])) / len(j[1]) * np.log((len(j[1]) - sum(j[1])) \
                                                                                / len(j[1])) - sum(
                    j[1]) / len(j[1]) * np.log(sum(j[1])) / len(j[1]))
            if entro < leastentro:
                leastentro = entro
                l = []
                for k in a:
                    l.append([min(k[0]), max(k[0]), np.log((sum(k[1]) / (len(k[1]) - sum(k[1]))) / (tt_bad / tt_good)),
                              sum(k[1]) / len(k[1])])
                    # print (sum(k[1]),len(k[1]))
    #4箱
    for m in range(10):
        y_pred = KMeans(n_clusters=5, random_state=m).fit_predict(x)
        a = [[[], []], [[], []], [[], []], [[], []], [[], []]]  
        for i in range(len(y_pred)):
            a[y_pred[i]][0].append(x[i][0])
            a[y_pred[i]][1].append(y[i])
        a = sorted(a, key=lambda x: sum(x[0]) / len(x[0]))
        if sum(a[0][1]) / len(a[0][1]) >= sum(a[1][1]) / len(a[1][1]) >= sum(a[2][1]) / len(a[2][1]) >= sum(a[3][1]) \
                / len(a[3][1]) >= sum(a[4][1]) / len(a[4][1]) or sum(a[0][1]) / len(a[0][1]) <= sum(a[1][1]) / len(
            a[1][1]) \
                <= sum(a[2][1]) / len(a[2][1]) <= sum(a[3][1]) / len(a[3][1]) <= sum(a[4][1]) / len(a[4][1]):
            entro = 0
            for k in a:
                entro = entro + (- (len(k[1]) - sum(k[1])) / len(k[1]) * np.log((len(k[1]) - sum(k[1])) \
                                                                                / len(k[1])) - sum(
                    k[1]) / len(k[1]) * np.log(sum(k[1])) / len(k[1]))
            if entro < leastentro:
                leastentro = entro
                l = []
                for k in a:
                    l.append([min(k[0]), max(k[0]), np.log((sum(k[1]) / (len(k[1]) - sum(k[1]))) / (tt_bad / tt_good)),
                              sum(k[1]) / len(k[1])])
                    # print (sum(k[1]),len(k[1]))
    if len(l) == 0:
        return 0
    else:
        dvars[item] = []
        scores[item] = []
        df_woe[item] = [0.0] * len(y_pred)
        print('\n', "Variable:", item, ": has ", len(l), "categories")
        for m in l:
            print("span=", [m[0], m[1]], ": WOE=", m[2], "; default rate=", m[3])
            dvars[item].append([m[0], m[2]])
            scores[item].append([[m[0], m[1]], m[2]])
            for i in range(len(y_pred)):
                if m[0] <= x[i] <= m[1]:
                    df_woe[item][i] = float(m[2])
        return 1

def woe3(y_pred, item, df, df_woe):
    total_bad = sum(df['target'])
    total_good = len(df['target']) - total_bad
    woe = []
    for i in range(3):  
        good, bad = 0, 0  
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                if df['target'][j] == 0:
                    good = good + 1
                else:
                    bad = bad + 1
        if bad == 0:
            bad = 1
        if good == 0:
            good = 1  
        woe.append((i, np.log((bad / good) / (total_bad / total_good))))
    df_woe[item] = [0.0] * len(y_pred)
    for i in range(len(y_pred)):
        for w in woe:
            if w[0] == y_pred[i]:
                df_woe[item][i] = float(w[1])
    return woe


def woe2(x_pred, item, df, df_woe):
    total_bad = sum(df['target'])
    total_good = len(df['target']) - total_bad
    X = np.array(df[item])
    y_pred = KMeans(n_clusters=2, random_state=1).fit_predict(x_pred)  
    woe = []
    judge = []
    for i in range(2):
        good, bad = 0, 0  
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                if df['target'][j] == 0:
                    good = good + 1
                else:
                    bad = bad + 1
        judge.append([i, bad / (bad + good)])
        if bad == 0:
            bad = 1
        if good == 0:
            good = 1  
        woe.append((i, np.log((bad / good) / (total_bad / total_good))))
    j0, j1 = [], []
    for k in range(len(y_pred)):
        if y_pred[k] == 0: j0.append(X[k])
        if y_pred[k] == 1: j1.append(X[k])
    jml = [[np.min(j0), np.max(j0)], [np.min(j1), np.max(j1)]]
    for l in range(2):
        judge[l].append(jml[l])
    judge = sorted(judge, key=lambda x: x[2])
    if judge[1][1] - judge[0][1] > 0:  # 违约率升序，则woe也升序
        woe = sorted(woe, key=lambda x: x[1])
    else:
        woe = sorted(woe, key=lambda x: x[1], reverse=True)
    dvars[item] = []
    scores[item] = []
    for i in range(2):
        # print("span=", judge[i][2], ": WOE=", woe[i][1], "; default rate=", judge[i][1])
        dvars[item].append([judge[i][2][0], woe[i][1]])
        scores[item].append([judge[i][2], woe[i][1]])
    df_woe[item] = [0.0] * len(y_pred)
    for i in range(len(y_pred)):
        for w in woe:
            if w[0] == y_pred[i]:
                df_woe[item][i] = float(w[1])
                
                
def calculate_woe(df):
    df_woe = pd.DataFrame() 
    for item in list(df)[:]: 
        X = np.array(df[item])  
        x_pred = []
        for it in X:
            x_pred.append([it])  
        flag = 0
        print(item, len(set(item)))
        if len(set(X)) > 4:
            res = woe_more(item, df, df_woe)
            if res == 1:
                continue
                flag = 1
        if 2 < len(set(X)) and flag == 0:
            for num in range(10):
                y_pred = KMeans(n_clusters=3, random_state=num).fit_predict(x_pred)  
                judge = []
                for i in range(3):  
                    good, bad = 0, 0  
                    for j in range(len(y_pred)):  
                        if y_pred[j] == i:
                            if df['target'][j] == 0:
                                good = good + 1
                            else:
                                bad = bad + 1
                    judge.append([i, bad / (bad + good)])
                j0, j1, j2 = [], [], []
                for k in range(len(y_pred)):
                    if y_pred[k] == 0: j0.append(X[k])
                    if y_pred[k] == 1: j1.append(X[k])
                    if y_pred[k] == 2: j2.append(X[k])
                jml = [[np.min(j0), np.max(j0)], [np.min(j1), np.max(j1)], [np.min(j2), np.max(j2)]]
                for l in range(3):
                    judge[l].append(jml[l])
                judge = sorted(judge, key=lambda x: x[2])
                if (judge[1][1] - judge[0][1]) * (judge[2][1] - judge[1][1]) >= 0:
                    woe = woe3(y_pred, item, df, df_woe)
                    print('\n', "Variable:", item, ": has 3 categories")
                    if judge[1][1] - judge[0][1] > 0:  # 违约率升序，则woe也升序
                        woe = sorted(woe, key=lambda x: x[1])
                    else:
                        woe = sorted(woe, key=lambda x: x[1], reverse=True)
                    dvars[item] = []
                    scores[item] = []
                    for i in range(3):
                        print("span=", judge[i][2], ": WOE=", woe[i][1], "; default rate=", judge[i][1])
                        dvars[item].append([judge[i][2][0], woe[i][1]])
                        scores[item].append([judge[i][2], woe[i][1]])
                    flag = 1
                    break
            if flag == 0:
                print('\n', "Variable:", item, ": has 2 categories")
                woe2(x_pred, item, df, df_woe)
        else:
            print('\n', "Variable:", item, ": must be 2 categories")
            woe2(x_pred, item, df, df_woe)
    df_woe['target'] = df['target']
    tar = df_woe['target']
    df_woe.drop(labels=['target'], axis=1, inplace=True)
    df_woe.insert(0, 'target', tar)
    return (df_woe)
                

def calculate_iv(df):  
    iv = []
    tar = df['target']
    tt_bad = sum(tar)
    tt_good = len(tar) - tt_bad
    for item in list(df)[1:]:
        x = df[item]
        st = set(x)
        s = 0.0
        for woe in st:
            tt = len(df[df[item] == woe]['target'])
            bad = sum(df[df[item] == woe]['target'])
            good = tt - bad
            s = s + float(bad / tt_bad - good / tt_good) * woe  
            #print("s:",s)
        iv.append([item, s])
    return sorted(iv, key=lambda x: x[1])

dvars = {}
scores = {}
df = pd.read_excel("German_credit.xlsx")
df=df.fillna(-999)

df_of_woe = calculate_woe(df)  

df_of_woe.to_excel("woe.xlsx")  


iv_list = calculate_iv(df_of_woe)
df_iv_list=pd.DataFrame(iv_list)
df_iv_list.to_excel("iv_list.xlsx")
                        
                        
                        
                        
                        




