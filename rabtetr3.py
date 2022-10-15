#1.3.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')

a=np.array([1, 4, 7])
ax.scatter(a[0],a[1],a[2])

b=np.array([-1, 2, 6])
ax.scatter(b[0],b[1],b[2])

c=np.array([0, -4, 0])
ax.scatter(c[0],c[1],c[2])

d=np.array([-3, 3, 0])
ax.scatter(d[0],d[1],d[2])

plt.show()

print(np.linalg.norm(a-d))
print(np.linalg.norm(b-d)**2)
print(np.linalg.norm(c-a, ord=np.inf))
print(np.linalg.norm(a-b,ord=1))

#1.3.2

T=np.zeros((5,5))
T+=np.arange(5)
print (T)

#2.3.1
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import seaborn as sns

iris = sns.load_dataset('iris')

X_train,X_test,y_train,y_test = train_test_split(
    iris.iloc[:,:-1],
    iris.iloc[:,-1],
    test_size= 0.15
)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train.head()
y_train.head()

model=KNeighborsClassifier(n_neighbors=15)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
y_pred

plt.figure(figsize=(16,7))
sns.scatterplot(x='petal_width',y='petal_length',data=iris,hue='species',s=70)
plt.xlabel('Длина лепестка,см')
plt.ylabel('Ширина лепестка,см')
plt.legend(loc=2)
plt.grid()

for i in range(len(y_test)):
    if np.array(y_test)[i] !=y_pred[i]:
        plt.scatter(X_test.iloc[i,3],X_test.iloc[i,2],color='red',s=150)

from sklearn.metrics import accuracy_score
print('accuracy:{accuracy_score(y_test,y_pred):.3}')
