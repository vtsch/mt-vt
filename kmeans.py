import os
from typing import Tuple
from bunch import Bunch
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import numpy as np
from plots import plot_centroids, plot_all_representations, run_umap

def kmeans(data: np.ndarray, config: Bunch) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Perform TS KMeans clustering on the data with the specified metric
    Args:
        data: data to cluster
        config: config file with parameters for kmeans
    Returns:
        centroids: centroids of the clusters
        labels: labels of the data given by kmeans
    '''
    # tskmeans takes data of shape (n_ts, sz, d)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    kmeans = TimeSeriesKMeans(n_clusters=config.n_clusters, metric=config.kmeans_metric, max_iter=10, random_state=0, n_init=5).fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def run_kmeans_and_plots(config: Bunch, embeddings: np.ndarray, true_labels: np.ndarray, experiment) -> np.ndarray:
    '''
    Run kmeans and plot the results
    Args:
        output: learned representations
        config: config file
        experiment: comet.ml experiment to log the figures
    Returns:
        centroids: centroids of the clusters
        labels: labels of the data given by kmeans
    '''
    centroids, kmeans_labels = kmeans(embeddings, config)
    plot_centroids(config, centroids, experiment)
    plot_all_representations(config, embeddings, kmeans_labels, experiment)
    run_umap(config, embeddings, true_labels, kmeans_labels, experiment)
    return kmeans_labels
