from sklearn.cluster import KMeans
import pandas as pd

class segment:
    def __init__(self, n_cluster = 4):
        self.kmeans = KMeans(n_clusters= n_cluster, random_state= 42)
        self.cluster_map = {
            0: 'Moderate Income, High Loan Burden',
            1: 'High Income, Low Default Risk',
            2: 'Moderate Income, Medium Risk',
            3: 'High Loan, Higher Default Risk'
        }

    def fit_predict(self, data):
        labels = self.kmeans.fit_predict(data)
        return labels
    
    def predict(self, data):
        labels = self.kmeans.predict(data)
        return labels
    def map_segment_labels(self, cluster_lables):
        return pd.Series(cluster_lables).map(self.cluster_map)
    
    

        