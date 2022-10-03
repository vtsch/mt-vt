import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import umap.umap_ as umap
import numpy as np

def plot_umap(embedding: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, name: str, experiment) -> None:    
    '''
    Plot UMAP embedding of the data
    Args:
        embedding: UMAP embedding of the data
        y_pred: predicted labels of the data
        y_real: real labels of the data
        name: name of the plot
        experiment: comet.ml experiment to log the figure
    ''' 
    colors  = [f"C{i}" for i in np.arange(0, 5)]
    cmap, norm = pltcolors.from_levels_and_colors(np.arange(0, 6), colors)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Clusters of Umap Embeddings of %s' %name)

    scatter1 = ax1.scatter(embedding[:,0], embedding[:,1], s=2, c=y_pred, cmap=cmap, norm=norm)
    ax1.set_title('Real')
    ax1.legend(*scatter1.legend_elements(), loc="upper left", title="Clusters")

    scatter2 = ax2.scatter(embedding[:,0], embedding[:,1], s=2, c=y_real, cmap=cmap, norm=norm)
    ax2.set_title('Predicted')
    ax2.legend(*scatter2.legend_elements(), loc="upper left", title="Clusters")
    experiment.log_figure(figure=plt, figure_name="umap_%s" %name)

def run_umap(output: np.ndarray, target: np.ndarray, kmeans_labels: np.ndarray, name: str, experiment) -> None:
    '''
    Run UMAP on the data and plot the embedding
    Args:
        output: output of the model
        target: target of the model
        kmeans_labels: labels of the kmeans clustering
        name: name of the plot
        experiment: comet.ml experiment to log the figure
    '''
    reducer = umap.UMAP(n_components=2)
    umap_emb = reducer.fit_transform(output)
    plot_umap(umap_emb, target, kmeans_labels, name, experiment)