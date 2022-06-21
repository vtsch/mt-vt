
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
import umap.umap_ as umap

def plot_centroids(centroids, n_clusters, title):
    for i in range(n_clusters):
        plt.plot(centroids[i])
    plt.title(title)
    plt.show()

def plot_umap(embedding, targets, c, title):
    plt.scatter(embedding[:,0], embedding[:,1], s=1, c=c, cmap='Spectral')
    plt.scatter(embedding[:,0], embedding[:,1], s=1, c=targets, cmap='Spectral')
    plt.title(title)
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
    plot_umap(umap_emb, target, kmeans_labels, "umap embedding %s encoder" %name)

def calculate_clustering_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    print('Clustering Accuracy: {:.2f}'.format(accuracy))
    print('Clustering ARI: {:.2f}'.format(ari))


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