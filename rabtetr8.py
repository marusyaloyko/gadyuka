# 1.1.1(1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

sns.set()

X = np.array([[5, 3], [10, 15], [15, 12], [24, 10],
              [30, 45], [85, 70], [71, 80], [60, 78],
              [55, 52], [80, 91]])

kmeans = KMeans(n_clusters=7, random_state=0)
clusters = kmeans.fit_predict(X.data)
print(kmeans.cluster_centers_.shape)

fig, ax = plt.subplots(2, 1, figsize=(20, 10))
centers = kmeans.cluster_centers_.reshape(1, 7, 2)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest',
               cmap=plt.cm.binary)

plt.show()

# 1.1.1(2)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

sns.set()

url = r'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534' \
      r'/iris.csv '
dataset = pd.read_csv(url)
arr = np.asarray(dataset.iloc[:, :-1])

kmeans = KMeans(n_clusters=150, random_state=0)
clusters = kmeans.fit_predict(arr.data)
print(kmeans.cluster_centers_.shape)

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
centers = kmeans.cluster_centers_.reshape(6, 10, 10)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest',
               cmap=plt.cm.binary)

plt.show()

# 1.1.2
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

url1 = r'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534' \
      r'/iris.csv '
dataset1 = pd.read_csv(url)
print(dataset1.head(), dataset1.shape)
data = dataset1.iloc[:, 1:4].values

plt.figure(figsize=(28, 12), dpi=180)
plt.figure(figsize=(10, 7))
plt.title("Iris Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))

cluster = AgglomerativeClustering(n_clusters=10,
                                  affinity='euclidean',
                                  linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], data[:, 2],
            c=cluster.labels_, cmap='rainbow')
plt.show()