from bunch import Bunch
import pandas as pd
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, rand_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier

# --- metrics for training ---

class Meter:
    def __init__(self):
        self.metrics = {}

    def update(self, x: torch.Tensor, y: torch.Tensor, phase: str, loss: torch.Tensor) -> None:
        '''
        Update metrics for a batch
        Parameters:
            x: input data
            y: true data
            phase: train or val
            loss: loss value
        '''
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.metrics[phase + '_loss'] += loss
        self.metrics[phase + '_mse'] += mean_squared_error(x,y)
        self.metrics[phase + '_mae'] += mean_absolute_error(x,y)
    
    def init_metrics(self, phase: str) -> None:
        '''
        Initialize metrics for a phase
        Parameters:
            phase: train or val
        '''
        self.metrics[phase + '_loss'] = 0
        self.metrics[phase + '_mse'] = 0
        self.metrics[phase + '_mae'] = 0
        
    def get_metrics(self):
        return self.metrics    
    
def get_knn_representation_score(embeddings, labels, experiment):
    '''
    Get representation score of embeddings
    Args:
        embeddings: embeddings
        data: data
        experiment: comet_ml experiment
    Returns:
        representation score
    '''
    # get 1-NN representation score (accuracy)
    clf = KNeighborsClassifier(n_neighbors=1).fit(embeddings, labels)
    preds = clf.predict(embeddings)
    # print sum of labels for each class
    print('Labels per class true:', np.unique(labels, return_counts=True))
    print('Labels per class predicted:', np.unique(preds, return_counts=True))
    representation_score = accuracy_score(labels, preds)
    experiment.log_metric('rep_accuracy', representation_score)
    return representation_score
        
# --- metrics for clustering evaluation ---

def calculate_clustering_scores(config: Bunch, y_true: np.ndarray, y_pred: np.ndarray, experiment) -> None:
    '''
    Calculate clustering scores of k-means 
    Args:
        y_true: true labels
        y_pred: predicted labels
        experiment: comet_ml experiment object
    '''
    accuracy = accuracy_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    print('Clustering Accuracy: {:.3f}'.format(accuracy))
    print('Clustering RI: {:.3f}'.format(ri))
    print('Clustering F1: {:.3f}'.format(f1))
    print('Confusion Matrix:')
    print(cm)
    metrics_df = pd.DataFrame({'accuracy': [accuracy], 'ri': [ri], 'f1': [f1]})
    experiment.log_metrics(metrics_df)
    experiment.log_confusion_matrix(y_true, y_pred, title = "Confusion Matrix")
    # save metrics and confusion matrix to csv
    metrics_df.to_csv(os.path.join(config.model_save_path, 'clustering_metrics.csv'))
    np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix.csv'), cm, delimiter=",")

def log_cluster_combinations(config: Bunch, true_labels: np.ndarray, kmeans_labels_old: np.ndarray, experiment) -> None:
    '''
    Log the cluster combinations, i.e. combine predicted labels to result in labels 0 and 1, test each combination and log the best one
    Args:
        config: config object
        true_labels: true labels
        kmeans_labels_old: predicted labels
        experiment: comet_ml experiment object
    '''
    # log cluster combinations
    if config.n_clusters == 3:
        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 1] = 0  
        kmeans_labels[kmeans_labels_old == 2] = 1
        f1_01 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_01 = confusion_matrix(true_labels, kmeans_labels)

        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 2] = 0
        f1_02 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_02 = confusion_matrix(true_labels, kmeans_labels)

        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 2] = 1
        f1_12 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_12 = confusion_matrix(true_labels, kmeans_labels)

        # find highest F1 score
        f1_scores = [f1_01, f1_02, f1_12]
        max_f1 = max(f1_scores)
        print('best F1 score: {:.3f}'.format(max_f1))
        #log F1 scores and log best one
        metrics_df = pd.DataFrame({'f1_01': [f1_01], 'f1_02': [f1_02], 'f1_12': [f1_12], 'f1_best': [max_f1]})
        experiment.log_metrics(metrics_df)
        # save metrics and confusion matrix to csv
        metrics_df.to_csv(os.path.join(config.model_save_path, 'metrics_3clusters.csv'))
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_01.csv'), cm_01, delimiter=",")
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_02.csv'), cm_02, delimiter=",")
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_12.csv'), cm_12, delimiter=",")
    
    elif config.n_clusters == 4:
        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 1] = 0  
        kmeans_labels[kmeans_labels_old == 2] = 0
        kmeans_labels[kmeans_labels_old == 3] = 1
        f1_012 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_012 = confusion_matrix(true_labels, kmeans_labels)

        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 1] = 0
        kmeans_labels[kmeans_labels_old == 2] = 1  
        kmeans_labels[kmeans_labels_old == 3] = 0
        f1_013 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_013 = confusion_matrix(true_labels, kmeans_labels)

        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 2] = 0  
        kmeans_labels[kmeans_labels_old == 3] = 0
        f1_023 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_023 = confusion_matrix(true_labels, kmeans_labels)

        kmeans_labels = kmeans_labels_old.copy()
        kmeans_labels[kmeans_labels_old == 2] = 1
        kmeans_labels[kmeans_labels_old == 3] = 1
        f1_123 = f1_score(true_labels, kmeans_labels, average="weighted")
        cm_123 = confusion_matrix(true_labels, kmeans_labels)

        # find highest F1 score
        f1_scores = [f1_012, f1_013, f1_023, f1_123]
        max_f1 = max(f1_scores)
        print('best F1 score: {:.3f}'.format(max_f1))

        #log F1 scores and log best one
        metrics_df = pd.DataFrame({'f1_012': [f1_012], 'f1_013': [f1_013], 'f1_023': [f1_023], 'f1_123': [f1_123], 'f1_best': [max_f1]})
        experiment.log_metrics(metrics_df)
        # save metrics and confusion matrix to csv
        metrics_df.to_csv(os.path.join(config.model_save_path, 'metrics_4clusters.csv'))
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_012.csv'), cm_012, delimiter=",")
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_013.csv'), cm_013, delimiter=",")
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_023.csv'), cm_023, delimiter=",")
        np.savetxt(os.path.join(config.model_save_path, 'confusion_matrix_123.csv'), cm_123, delimiter=",")
    
    else:
        print('No cluster combinations logged')


    