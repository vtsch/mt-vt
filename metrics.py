import torch
import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, adjusted_mutual_info_score, mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score

def calculate_clustering_scores(y_true, y_pred, experiment):
    accuracy = accuracy_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    mi = mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print('Clustering Accuracy: {:.3f}'.format(accuracy))
    print('Clustering RI: {:.3f}'.format(ri))
    print('Clustering ARI: {:.3f}'.format(ari))
    print('Clustering MI: {:.3f}'.format(mi))
    print('Clustering AMI: {:.3f}'.format(ami))
    print('Clustering NMI: {:.3f}'.format(nmi))
    experiment.log_metrics({"Clustering Accuracy": accuracy, "Clustering RI": ri, "Clustering ARI": ari, "Clustering MI": mi, "Clustering AMI": ami, "Clustering NMI": nmi})


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