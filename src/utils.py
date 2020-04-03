import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score

def externalValidation(truthClusters, predictedClusters):
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    scores = {}
    scores['_rand_index'] = adjusted_rand_score(truthClusters, predictedClusters)
    scores['_homogeneity_score'] = metrics.homogeneity_score(truthClusters, predictedClusters)
    scores['_purity_score'] = purity_score(truthClusters, predictedClusters)
    scores['_adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(truthClusters, predictedClusters)
    scores['_fowlkes_mallows_score'] = metrics.fowlkes_mallows_score(truthClusters, predictedClusters)
    return scores


def internalValidation(data, clusters):
    scores = {}
    """
    The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    Scores around zero indicate overlapping clusters.
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    """
    scores['silhouette_score'] = metrics.silhouette_score(data, clusters, metric='euclidean')
    """
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    The score is fast to compute
    """
    scores['calinski_harabaz_score'] = metrics.calinski_harabasz_score(data, clusters)
    """
    Zero is the lowest possible score. Values closer to zero indicate a better partition.
    The Davies-Boulding index is generally higher for convex clusters than other concepts of clusters, 
    such as density based clusters like those obtained from DBSCAN.
    """
    scores['davies_bouldin_score'] = metrics.davies_bouldin_score(data, clusters)
    return scores
[]