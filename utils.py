
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import torch
import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, adjusted_mutual_info_score, mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score
import umap.umap_ as umap

def plot_centroids(centroids, n_clusters, title):
    for i in range(n_clusters):
        plt.plot(centroids[i])
    plt.title(title)
    plt.show()

def plot_umap(embedding, y_pred, y_real, name):     
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
    plt.show()

def plot_loss(history, title):
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def run_umap(output, target, kmeans_labels, name):
    reducer = umap.UMAP(n_components=2)
    umap_emb = reducer.fit_transform(output)
    plot_umap(umap_emb, target, kmeans_labels, name)
    

def calculate_clustering_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    mi = mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    vmi = v_measure_score(y_true, y_pred)
    print('Clustering Accuracy: {:.3f}'.format(accuracy))
    print('Clustering RI: {:.3f}'.format(ri))
    print('Clustering ARI: {:.3f}'.format(ari))
    print('Clustering MI: {:.3f}'.format(mi))
    print('Clustering AMI: {:.3f}'.format(ami))
    print('Clustering NMI: {:.3f}'.format(nmi))
    print('Clustering VMI: {:.3f}'.format(vmi))


class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))
    
    def update(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        self.metrics['accuracy'] += accuracy_score(x,y)
        self.metrics['f1'] += f1_score(x,y,average='macro')
        #self._compute_cm(x, y)
        
    def _compute_cm(self, x, y):
        for prob, target in zip(x, y):
            if prob == target:
                self.confusion[target][target] += 1
            else:
                self.confusion[target][prob] += 1
    
    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        
    def get_metrics(self):
        return self.metrics
    
    def get_confusion_matrix(self):
        return self.confusion