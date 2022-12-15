# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:51:03 2022

@author: ASUS
"""

from sklearn.decomposition import PCA
import numpy as np

model=PCA(n_components=1)
#3个特征，6个样本
x=np.array([[1,2,4,3,8,2],
           [2,3,5,6,8,2],
           [1,4,5,8,1,3]])
model.fit(x.T)
c=model.transform(x.T)
c

#做一次以0为中心化
x_mean=np.repeat(x.mean(axis=1).reshape(-1,1),repeats=6,axis=1)
x_new=x-x_mean

b=(1/x_new.shape[1])*x_new@x_new.T#3x6再来乘以6x3
values,vector=np.linalg.eig(b)
values=values.astype(float)
vector=vector.astype(float)

vector[:,1]@x_new#乘以的是中心化的变量

#%%
from sklearn.preprocessing import StandardScaler
model=StandardScaler()
x_new=model.fit_transform(x.T).T
#此时还是3x6
cov_=x_new@x_new.T*(1/x_new.shape[1])
lambda_,vector=np.linalg.eig(cov_)
lambda_
vector
x_new.T@vector[:,1]
#%%
'''
构建特征值和特征向量的组合
'''
a=[]
for i in range(len(values)):
    m=(values[i],vector[:,i])
    a.append(m)
a.sort(key=lambda x : x[0],reverse=True)

total=values.sum()
for i in sorted(values,reverse=True):
    prob=i/total
    print(prob)


np.cumsum(sorted(values,reverse=True))/total


a=np.mat([[1,2],
           [3,4]])

b=np.mat([[2,1],
           [4,3]])

np.dot(a,b)
a*b

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from math import sqrt
model=load_iris()
x=model.data
x=x[:,:2]

#开始进行k-means聚类
def category_point(sample,initial_point,category):#一次迭代生成的分类值
    x_category=np.zeros(sample.shape[0])#存储分类值的
    mindistance=10000#假设一个最小距离
    for i in range(sample.shape[0]):#遍历每一个样本
        for k in range(category):#遍历质心点个数
            distance=sqrt(sum(pow(x[i]-initial_point[k],2)))
            if distance<mindistance:
                mindistance=distance
                x_category[i]=k#更新每个样本点的分类标签

    return x_category


def change_original(x,original,cluster):
    x_class=np.zeros(x.shape[0])#存储分类值的
    row=np.zeros((cluster,cluster))#来存储质心点的
    for iters in range(10000):
        if (x_class!=category_point(x, original,cluster)).sum()!=0:#与第一个对比，如果不等于则把原始的分类改变
            x_class=category_point(x, original,cluster)
            
            for k in range(cluster):#更新质心
                row[k]=x[np.where(category_point(x,original,cluster)==k)].mean(axis=0)
            original=row#每次改变聚合中心点
                
    return row,x_class
        

mean_point=np.array([[6,3],
                     [5,3]])

row,x_category=change_original(x, mean_point, 2)


plt.scatter(x[:,0],x[:,1],c=x_category)

#%%
from sklearn.cluster import KMeans

model=KMeans(n_clusters=2)
model.fit(x)
m = model.labels_ 
plt.scatter(x[:,0],x[:,1],c=m)        
model.cluster_centers_

y=np.random.rand(150)
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=3,include_bias=False)
x_2_fit=pf.fit_transform(x)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_2_fit,y)
model.intercept_
model.coef_












