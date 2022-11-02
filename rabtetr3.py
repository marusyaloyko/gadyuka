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
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

iris = sns.load_dataset('iris')

x_tr, x_t, y_tr, y_t = train_test_split(
    iris.iloc[:, :-1],
    iris.iloc[:, -1],
    test_size=0.15
)

k = 1  # 5 and 10
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_tr, y_tr)

y_pr = model.predict(x_t)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=iris,
    x='petal_width', y='petal_length',
    hue='species',
    s=70
)
plt.xlabel('Длина лепестка, см')
plt.ylabel('Ширина лепестка, см')
plt.legend(loc=2)
plt.grid()

for i in range(len(y_t)):
    if np.array(y_t)[i] != y_pr[i]:
        plt.scatter(x_t.iloc[i, 3], x_t.iloc[i, 2], color='red', s=150)

print(f'accuracy: {accuracy_score(y_t, y_pr):.3}')

plt.show()

# 3.3.2
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

da = pd.DataFrame({"тип темперамента": ["холерик", "сангвиник", "меланхолик", "флегматик",]})

scale_mapper = {"холерик": 1, "сангвиник": 2, "меланхолик": 3, "флегматик": 4}

da["тип темперамента"].replace(scale_mapper)
print(da, "\n")

dic = [{"холерик": 2, "сангвиник": 7}, {"меланхолик": 1, "флегматик": 9}]

d = DictVectorizer(sparse=False)

features = d.fit_transform(dic)

print(features)
