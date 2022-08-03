import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error

def calculate_clustering_scores(y_true, y_pred, experiment):
    accuracy = accuracy_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    mi = mutual_info_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print('Clustering Accuracy: {:.3f}'.format(accuracy))
    print('Clustering RI: {:.3f}'.format(ri))
    print('Clustering ARI: {:.3f}'.format(ari))
    print('Clustering MI: {:.3f}'.format(mi))
    print('Clustering NMI: {:.3f}'.format(nmi))
    print('Confusion Matrix:')
    print(cm)
    experiment.log_metrics(pd.DataFrame({'accuracy': [accuracy], 'ri': [ri], 'ari': [ari], 'mi': [mi], 'nmi': [nmi]}))
    experiment.log_confusion_matrix(y_true, y_pred, title = "Confusion Matrix")


class Meter:
    def __init__(self):
        self.metrics = {}

    def update(self, x, y, phase, loss):
        #x = np.argmax(x.detach().cpu().numpy(), axis=1)
        #y = y.detach().cpu().numpy()
        self.metrics[phase + '_loss'] += loss
        self.metrics[phase + '_mse'] += mean_squared_error(x,y)
        self.metrics[phase + '_r2'] += r2_score(x,y)
    
    def init_metrics(self, phase):
        self.metrics[phase + '_loss'] = 0
        self.metrics[phase + '_mse'] = 0
        self.metrics[phase + '_r2'] = 0
        
    def get_metrics(self):
        return self.metrics
    