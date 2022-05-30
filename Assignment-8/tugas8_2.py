import matplotlib.pyplot as plt
import numpy as numpy
import pandas as pd 
from sklearn.cluster import KMeans

dataset = pd.read_csv('C:\Users\acer\OneDrive\Desktop\Data.csv')
dataset.keys()

dataku = pd.DataFrame(dataset)
dataku.head()

X = np.asarray(dataset)
print(X)

plt.scatter(X[:,0],X[:,1], label='True Position')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafik Penyebaran Data Konsumen")
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.cluster_centers_)

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap ='rainbow')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafik Hasil Klasterisasi Data Gaji dan Pengeluaran Konsumen")
plt.show()

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap ='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color = 'black')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafik Hasil Klasterisasi Data Gaji dan Pengeluaran Konsumen")
plt.show()