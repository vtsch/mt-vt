from typing import Tuple
from sklearn import cluster
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import numpy as np

def plot_datapoints_of_cluster(data: np.ndarray, labels: np.ndarray, cluster_id: int, label_color_map: list, title: str, experiment) -> None:
    '''
    Parameters:
        data: data to plot
        labels: labels of the data
        cluster_id: id of the cluster to plot
        label_color_map: list of colors for each cluster
        title: title of the plot
        experiment: comet.ml experiment to log the figure
    '''
    datapoints = data[labels == cluster_id]
    print("Cluster %d has %d datapoints predicted" %(cluster_id, datapoints.shape[0]))

    fig1 = plt.figure("%s cluster %d" %(title, cluster_id))
    #plot data points with label = cluster_id
    for i in range(datapoints.shape[0]):
        plt.plot(datapoints[i], c=label_color_map[cluster_id])
        plt.ylim(-2,10)
    plt.title("%s datapoints of cluster %d" %(title, cluster_id))
    plt.legend(labels=[cluster_id], loc='upper left', title="Clusters")
    experiment.log_figure(figure=fig1, figure_name="datapoints_%s_cluster_%d" %(title, cluster_id))

    fig2 = plt.figure("%s boxplot cluster %d" %(title, cluster_id))
    # create boxplot of datapoints of cluster_id for each timestep
    plt.boxplot(datapoints, widths=0.6, patch_artist=True, boxprops=dict(facecolor=label_color_map[cluster_id]))
    plt.ylim(-2,10)
    plt.title("%s boxplot of cluster %d" %(title, cluster_id))
    plt.legend(labels=[cluster_id], loc='upper left', title="Clusters")
    experiment.log_figure(figure=fig2, figure_name="boxplot_%s_cluster_%d" %(title, cluster_id))


def plot_datapoints(data: np.ndarray, labels: np.ndarray, title: str, experiment) -> None:
    '''
    Parameters:
        data: data to plot
        labels: labels of the data
        title: title of the plot
        experiment: comet.ml experiment to log the figure
    '''
    fig0 = plt.figure("%s" %title)
    #set colors for each cluster label
    label_color_map = ['#3cb44b', '#4363d8', '#ffe119', '#f58231', '#911eb4']
    for i in range(len(data)):
        #plot data points with different colors for each cluster
        plt.plot(data[i], c=label_color_map[labels[i]])
        plt.ylim(-2,10)
    plt.title(title)
    #plt.ylim(0,4)
    plt.legend(['%d' %i for i in range(np.unique(labels).shape[0])], loc='upper left', title="Clusters")
    experiment.log_figure(figure=fig0, figure_name="datapoints_%s" %title)

    for i in range(np.unique(labels).shape[0]):
        plot_datapoints_of_cluster(data, labels, i, label_color_map, title, experiment)


def plot_centroids(centroids: np.ndarray, n_clusters: int, title: str, experiment) -> None:
    '''
    Parameters:
        centroids: centroids to plot
        n_clusters: number of clusters
        title: title of the plot
        experiment: comet.ml experiment to log the figure
    '''
    for i in range(n_clusters):
        plt.plot(centroids[i])
    plt.title(title)
    plt.legend(['%d' %i for i in range(n_clusters)], loc='upper left', title="Clusters")
    experiment.log_figure(figure=plt, figure_name="centroids_%s" %title)

def kmeans(data: np.ndarray, n_clusters: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parameters:
        data: data to cluster
        n_clusters: number of clusters
        metric: metric to use for clustering
    Returns:
        centroids: centroids of the clusters
        labels: labels of the data given by kmeans
    '''
    # tskmeans takes data of shape (n_ts, sz, d)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=10, random_state=0, n_init=5).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def run_kmeans_and_plots(output: np.ndarray, config: dict, experiment) -> np.ndarray:
    '''
    Parameters:
        output: learn representation of the model
        config: config file
        experiment: comet.ml experiment to log the figures
    Returns:
        centroids: centroids of the clusters
        labels: labels of the data given by kmeans
    '''
    centroids, kmeans_labels = kmeans(output, config.n_clusters, config.kmeans_metric)
    plot_centroids(centroids, config.n_clusters, "%s kmeans centroids %s" %(config.kmeans_metric, config.experiment_name), experiment)
    plot_datapoints(output, kmeans_labels, config.experiment_name, experiment)
    return kmeans_labels
