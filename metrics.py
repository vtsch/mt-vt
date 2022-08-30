import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, rand_score, adjusted_rand_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_clustering_scores(y_true, y_pred, experiment):
    accuracy = accuracy_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print('Clustering Accuracy: {:.3f}'.format(accuracy))
    print('Clustering RI: {:.3f}'.format(ri))
    print('Clustering ARI: {:.3f}'.format(ari))
    print('Clustering F1: {:.3f}'.format(f1))
    print('Confusion Matrix:')
    print(cm)
    experiment.log_metrics(pd.DataFrame({'accuracy': [accuracy], 'ri': [ri], 'ari': [ari], 'f1': [f1]}))
    experiment.log_confusion_matrix(y_true, y_pred, title = "Confusion Matrix")

class Meter:
    def __init__(self):
        self.metrics = {}

    def update(self, x, y, phase, loss):
        #x = np.argmax(x.detach().cpu().numpy(), axis=1)
        #y = y.detach().cpu().numpy()
        self.metrics[phase + '_loss'] += loss
        self.metrics[phase + '_mse'] += mean_squared_error(x,y)
        self.metrics[phase + '_mae'] += mean_absolute_error(x,y)
        self.metrics[phase + '_r2'] += r2_score(x,y)
    
    def init_metrics(self, phase):
        self.metrics[phase + '_loss'] = 0
        self.metrics[phase + '_mse'] = 0
        self.metrics[phase + '_mae'] = 0
        self.metrics[phase + '_r2'] = 0
        
        
    def get_metrics(self):
        return self.metrics
    