from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

def plot_centroids(centroids, n_clusters, title, experiment):
    for i in range(n_clusters):
        plt.plot(centroids[i])
    plt.title(title)
    plt.legend(['%d' %i for i in range(n_clusters)], loc='upper left', title="Clusters")
    experiment.log_figure(figure=plt, figure_name="centroids_%s" %title)

def kmeans(data, n_clusters, metric):
    # tskmeans takes data of shape (n_ts, sz, d)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=20, random_state=0, n_init=5).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def run_kmeans(output, config, experiment):
    centroids, kmeans_labels = kmeans(output, config.n_clusters, config.metric)
    plot_centroids(centroids, config.n_clusters, "%s kmeans centroids %s" %(config.metric, config.experiment_name), experiment)
    return kmeans_labels

def run_kmeans_only(data, n_clusters, metric):
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=20, random_state=0, n_init=5).fit(data)
    kmeans_labels = kmeans.predict(data)
    return kmeans_labels