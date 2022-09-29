import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, rand_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_clustering_scores(y_true, y_pred, experiment):
    accuracy = accuracy_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    print('Clustering Accuracy: {:.3f}'.format(accuracy))
    print('Clustering RI: {:.3f}'.format(ri))
    print('Clustering F1: {:.3f}'.format(f1))
    print('Confusion Matrix:')
    print(cm)
    experiment.log_metrics(pd.DataFrame({'accuracy': [accuracy], 'ri': [ri], 'f1': [f1]}))
    experiment.log_confusion_matrix(y_true, y_pred, title = "Confusion Matrix")

def log_cluster_combinations(true_labels, kmeans_labels_old, experiment):
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

    #log F1 scores and log best one
    experiment.log_metrics(pd.DataFrame({'f1_01': [f1_01], 'f1_02': [f1_02], 'f1_12': [f1_12]}))
    if f1_01 > f1_02 and f1_01 > f1_12:
        experiment.log_metrics(pd.DataFrame({'f1_best': [f1_01]}))
    elif f1_02 > f1_01 and f1_02 > f1_12:
        experiment.log_metrics(pd.DataFrame({'f1_best': [f1_02]}))
    else:
        experiment.log_metrics(pd.DataFrame({'f1_best': [f1_12]}))
    
    #print all f1 scores and confusion matrices
    print('F1 01: {:.3f}'.format(f1_01))
    print('Confusion Matrix 01:')
    print(cm_01)
    print('F1 02: {:.3f}'.format(f1_02))
    print('Confusion Matrix 02:')
    print(cm_02)
    print('F1 12: {:.3f}'.format(f1_12))
    print('Confusion Matrix 12:')
    print(cm_12)


class Meter:
    def __init__(self):
        self.metrics = {}

    def update(self, x, y, phase, loss):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.metrics[phase + '_loss'] += loss
        self.metrics[phase + '_mse'] += mean_squared_error(x,y)
        self.metrics[phase + '_mae'] += mean_absolute_error(x,y)
    
    def init_metrics(self, phase):
        self.metrics[phase + '_loss'] = 0
        self.metrics[phase + '_mse'] = 0
        self.metrics[phase + '_mae'] = 0
        
    def get_metrics(self):
        return self.metrics
    