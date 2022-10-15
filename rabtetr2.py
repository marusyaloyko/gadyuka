import numpy as np
#1.3.1

arr=np.array([[1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1]])

print(arr)

#1.3.2
N=np.array([np.arange(5) for i in range(5)])
print(N)

#1.3.3
P=np.random.rand(3,3,3)
print(P)

#1.3.4
K=np.array([[1,1,1],
           [1,0,1],
           [1,1,1]])
print(K)

#1.3.5
T=np.array([[1,2,3],
[5,8,4],
[5,7,3]])
print(np.sort(T, axis=None)[::-1].reshape(3,3))

#1.3.6
M=np.random.rand(4,4)
print(M.shape,M.size)

#2.3.1
import pandas as pd
a=[1,3,1]
b=[5,0,1]
s1=pd.Series(a)
s2=pd.Series(b)
square=np.square(s1-s2)
sum=np.sum(square)
print(np.sqrt(sum))

#2.3.2
ur1='https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'
dataframe=pd.read_csv(ur1)
r=dataframe.head(8)
print(r)

#2.3.3
dataframe.tail(8)
dataframe.shape
dataframe.describe
print(dataframe.shape,dataframe.describe)

m=dataframe.iloc[1:2]
print(m)

print(dataframe[dataframe['PClass']=='1st'].head(1))


#3.3.2
ur2='https://raw.githubusercontent.com/akmand/datasets/master/iris.csv'
dataframe=pd.read_csv(ur2)
print(dataframe)

import numpy as np
from sklearn import preprocessing
dataframe['sepal length(cm)'].min(),\
dataframe['sepal length(cm)'].max()
