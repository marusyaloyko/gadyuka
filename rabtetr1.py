#1.3
from statistics import mean

x=5<=2
A={1,3,7,8}
B={2,4,5,10,'apple'}
C=A&B
df='антонина антонова',34,'ж'
z='type'
D=[1,'title',2,'content']

print(x,'|',type(x),'\n',A,'|',type(A),'\n',B,'|',type(B),'\n',C,'|',type(C),'\n',df,'|',type(df),'\n',z,'|',type(z),'\n',D,'|',type(D))

#2.3

x=-5.5
if -5<x<5:
    print("x принадлежит (-5;5)")
elif x>5:
    print("x принадлежит (5;inf)")
else:
    print("х принадлежит (-inf;-5)")

#3.3.1

x=10
while x>=1:
    print(x)
    x-=1

#3.3.2

models=['признак1','признак2','признак3','признак4']
for models in models:
    print(models)

#3.3.3

a=2
while a<=15:
    print(a)
    a+=1

#3.3.4

for i in range(105, 5, -25):
    print(i)

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#3.3.5

l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(1, int(len(l)/2), 2):
    if not len(l) % 2:
        l[i], l[-i] = l[-i], l[i]
    else:
        l[i], l[-i-1] = l[-i-1], l[i]

print(l)

#4.3.1

import matplotlib.pyplot as plt
import numpy as np
import random
for i in range(4):
    print(random.random())
t=np.mean(random.random())
m=np.median(random.random())
print(t)
print(m)
plt.grid()





