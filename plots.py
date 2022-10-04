import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import umap.umap_ as umap
import numpy as np
import os
from typing import Tuple
from bunch import Bunch
import matplotlib.pyplot as plt

def plot_datapoints_of_cluster(config: Bunch, data: np.ndarray, labels: np.ndarray, cluster_id: int, label_color_map: list, experiment) -> None:
    '''
    Plot datapoints of a specific cluster, plot the TS as lines and boxplots
    Args:
        config: config of the experiment
        data: data to plot
        labels: labels of the data
        cluster_id: id of the cluster to plot
        label_color_map: list of colors for each cluster
        experiment: comet.ml experiment to log the figure
    '''
    datapoints = data[labels == cluster_id]
    print("Cluster %d has %d datapoints predicted" %(cluster_id, datapoints.shape[0]))

    title1 = "datapoints_cluster_%d_%s_%s" %(cluster_id, config.experiment_name, config.pos_enc)
    fig1 = plt.figure(title1)
    #plot data points with label = cluster_id
    for i in range(datapoints.shape[0]):
        plt.plot(datapoints[i], c=label_color_map[cluster_id])
        plt.ylim(-2,10)
    plt.title(title1)
    plt.legend(labels=[cluster_id], loc='upper left', title="Clusters")
    experiment.log_figure(figure=fig1, figure_name=title1)
    plt.savefig(os.path.join(config.model_save_path, "%s.png" %title1))

    title2 = "boxplot_cluster_%d_%s_%s" %(cluster_id, config.experiment_name, config.pos_enc)
    fig2 = plt.figure(title2)
    # create boxplot of datapoints of cluster_id for each timestep
    plt.boxplot(datapoints, widths=0.6, patch_artist=True, boxprops=dict(facecolor=label_color_map[cluster_id]))
    plt.ylim(-2,10)
    plt.title(title2)
    plt.legend(labels=[cluster_id], loc='upper left', title="Clusters")
    experiment.log_figure(figure=fig2, figure_name=title2)
    plt.savefig(os.path.join(config.model_save_path, "%s.png" %title2))


def plot_datapoints(config: Bunch, data: np.ndarray, labels: np.ndarray, experiment) -> None:
    '''
    Plot datapoints of each cluster together
    Args:
        config: config of the experiment
        data: data to plot
        labels: labels of the data
        experiment: comet.ml experiment to log the figure
    '''
    title = "datapoints_%s" %config.experiment_name
    fig0 = plt.figure(title)
    #set colors for each cluster label
    label_color_map = ['#3cb44b', '#4363d8', '#ffe119', '#f58231', '#911eb4']
    for i in range(len(data)):
        #plot data points with different colors for each cluster
        plt.plot(data[i], c=label_color_map[labels[i]])
        plt.ylim(-2,10)
    plt.title(title)
    #plt.ylim(0,4)
    plt.legend(['%d' %i for i in range(np.unique(labels).shape[0])], loc='upper left', title="Clusters")
    experiment.log_figure(figure=fig0, figure_name=title)
    plt.savefig(os.path.join(config.model_save_path, "%s.png" %title))

    for i in range(np.unique(labels).shape[0]):
        plot_datapoints_of_cluster(config, data, labels, i, label_color_map, experiment)


def plot_centroids(config: Bunch, centroids: np.ndarray, experiment) -> None:
    '''
    Plot centroids of the clusters
    Args:
        config: config of the experiment
        centroids: centroids to plot
        experiment: comet.ml experiment to log the figure
    '''
    title = "%s kmeans centroids %s" %(config.kmeans_metric, config.experiment_name)
    for i in range(config.n_clusters):
        plt.plot(centroids[i])
    plt.title(title)
    plt.legend(['%d' %i for i in range(config.n_clusters)], loc='upper left', title="Clusters")
    experiment.log_figure(figure=plt, figure_name=title)
    plt.savefig(os.path.join(config.model_save_path, "%s.png" %title))
        

def plot_umap(config: Bunch, umap_embedding: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, experiment) -> None:    
    '''
    Plot UMAP embedding of the data
    Args:
        config: config of the experiment
        embedding: UMAP embedding of the data
        y_pred: predicted labels of the data
        y_real: real labels of the data
        experiment: comet.ml experiment to log the figure
    ''' 
    colors  = [f"C{i}" for i in np.arange(0, 5)]
    cmap, norm = pltcolors.from_levels_and_colors(np.arange(0, 6), colors)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Clusters of Umap Embeddings of %s' %config.experiment_name)

    scatter1 = ax1.scatter(umap_embedding[:,0], umap_embedding[:,1], s=2, c=y_pred, cmap=cmap, norm=norm)
    ax1.set_title('Real')
    ax1.legend(*scatter1.legend_elements(), loc="upper left", title="Clusters")

    scatter2 = ax2.scatter(umap_embedding[:,0], umap_embedding[:,1], s=2, c=y_real, cmap=cmap, norm=norm)
    ax2.set_title('Predicted')
    ax2.legend(*scatter2.legend_elements(), loc="upper left", title="Clusters")

    experiment.log_figure(figure=plt, figure_name="umap_%s" %config.experiment_name)
    plt.savefig(os.path.join(config.model_save_path, "umap_%s.png" %config.experiment_name))

def run_umap(config: Bunch, output: np.ndarray, true_labels: np.ndarray, kmeans_labels: np.ndarray, experiment) -> None:
    '''
    Run UMAP on the data
    Args:
        config: config of the experiment
        output: output of the model
        true_labels: real labels of the data
        kmeans_labels: predicted labels of the data
        experiment: comet.ml experiment to log the figure
    '''
    umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation', n_components=2).fit_transform(output)
    plot_umap(config, umap_embedding, kmeans_labels, true_labels, experiment)

