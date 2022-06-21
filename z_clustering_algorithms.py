import random
import numpy as np
from sklearn.cluster import KMeans
from z_embeddings import umap_embedding
from z_utils import plot_centroids, plot_umap

def sklearnkmeans(data, n_clusters):
    kmeans = KMeans(n_clusters).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_

    return centroids, labels

def DTWDistance(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def k_means_dtw(data,num_clust,num_iter,w=5):
    centroids=random.sample(list(data),num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print(counter)
        assignments={}
        labels = []

        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                cur_dist=DTWDistance(i,j,w)
                if cur_dist<min_dist:
                    min_dist=cur_dist
                    closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]
            labels.append(closest_clust)
    
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    
    return centroids, labels

def run_kmeans(output, n_clusters, name):
    centroids, kmeans_labels = sklearnkmeans(output, n_clusters)
    plot_centroids(centroids, n_clusters, "kmeans centroids %s" %name)

    umap_emb = umap_embedding(output)
    plot_umap(umap_emb, kmeans_labels, "umap embedding %s encoder" %name)

    return kmeans_labels