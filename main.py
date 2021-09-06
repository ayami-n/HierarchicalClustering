import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

def Hierarchical_c():

   # import data
   data = pd.read_csv('Mall_Customers.csv')
   x = data.iloc[:, [3, 4]].values

   # Using the dendrogram to find the optimal number of clusters
   dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
   plt.title('Dendrogram')
   plt.xlabel('Customer')
   plt.ylabel('Euclidean distances')
   plt.show()

   # Training the Hierarchical Clustering model on the data
   hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
   y_hc = hc.fit_predict(x)

   # Visualising the clusters
   plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c='yellow', label='Cluster 1')
   plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c='red', label='Cluster 2')
   plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
   plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c='pink', label='Cluster 4')
   plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, c='blue', label='Cluster 5')
   plt.title('Clusters of customers')
   plt.xlabel('Annual Income (k$)')
   plt.ylabel('Spending Score (1-100)')
   plt.legend()
   plt.show()

if __name__ == '__main__':
    Hierarchical_c()

