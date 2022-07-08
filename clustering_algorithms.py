from tslearn.clustering import TimeSeriesKMeans
from utils import plot_centroids

def kmeans(data, n_clusters, metric):
    # tskmeans takes data of shape (n_ts, sz, d)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=5, random_state=0, n_init=5).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def run_kmeans(output, n_clusters, metric, name):
    centroids, kmeans_labels = kmeans(output, n_clusters, metric)
    plot_centroids(centroids, n_clusters, "%s kmeans centroids %s" %(metric, name))
    return kmeans_labels



