# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:57:09 2021

@author: EXTRA
"""

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=100, n_features=2, centers=[(0,0),(3,3)], cluster_std=2,
                  random_state=1)
plt.figure(1)
plt.scatter(X[:,0],X[:,1])

df_X = pd.DataFrame(data = X,columns=['x1','x2'])
df_y = pd.DataFrame(data = y, columns = ['y'])
df_Xy = pd.concat([df_X,df_y],axis=1)
c0 = df_Xy[df_Xy['y']==0]
c1 = df_Xy[df_Xy['y']==1]
plt.figure(2)
plt.scatter(c0['x1'],c0['x2'], color='r')
plt.scatter(c1['x1'],c1['x2'], color='g')

in0 = np.where(y==0)
index0 = in0[0]
in1 = np.where(y==1)
index1 = in1[0]
plt.figure(3)
plt.scatter(X[index0,0], X[index0,1], color='r')
plt.scatter(X[index1,1], X[index1,1], color='g')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model=LinearSVC(C=1, max_iter=1000)
model.fit(X_train, y_train)
y_estime_apprentissage = model.predict(X_train)
acc_apprentissage = accuracy_score(y_train, y_estime_apprentissage)
print(acc_apprentissage)

y_estime_test = model.predict(X_test)
acc_test = accuracy_score(y_test, y_estime_test)
print(acc_test)
