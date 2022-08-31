from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import numpy as np

def plot_datapoints(data, labels, title, experiment):
    #set colors for each cluster label
    label_color_map = ['#3cb44b', '#4363d8', '#ffe119', '#f58231', '#911eb4']
    for i in range(len(data)):
        #plot data points with different colors for each cluster
        plt.plot(data[i], c=label_color_map[labels[i]])
    plt.title(title)
    plt.legend(['%d' %i for i in range(np.unique(labels).shape[0])], loc='upper left', title="Clusters")
    experiment.log_figure(figure=plt, figure_name="datapoints_%s" %title)

    #plot only datapoints with label 0
    plt.plot(data[labels == 0], c='#3cb44b')
    plt.title("%s datapoints with label 0" %title)
    experiment.log_figure(figure=plt, figure_name="datapoints_%s_label_0" %title)

    #plot only datapoints with label 1
    plt.plot(data[labels == 1], c='#4363d8')
    plt.title("%s datapoints with label 1" %title)
    experiment.log_figure(figure=plt, figure_name="datapoints_%s_label_1" %title)

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

def run_kmeans_only(data, config):
    data.reshape(data.shape[0], data.shape[1], 1)
    kmeans = TimeSeriesKMeans(n_clusters=config.n_clusters, metric=config.metric, max_iter=20, random_state=0, n_init=5).fit(data)
    kmeans_labels = kmeans.predict(data)
    return kmeans_labels