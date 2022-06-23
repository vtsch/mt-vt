from tslearn.clustering import TimeSeriesKMeans
from utils import plot_centroids

def euclkmeans(data, n_clusters):
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=5).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def dtwkmeans(data, n_clusters):
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=5).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def run_kmeans(output, n_clusters, name):
    centroids, kmeans_labels = euclkmeans(output, n_clusters)
    plot_centroids(centroids, n_clusters, "eucl kmeans centroids %s" %name)
    return kmeans_labels

def run_dtw_kmeans(output, n_clusters, name):
    centroids, kmeans_labels = dtwkmeans(output, n_clusters)
    plot_centroids(centroids, n_clusters, " dtw kmeans centroids %s" %name)
    return kmeans_labels




