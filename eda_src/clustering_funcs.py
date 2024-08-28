from sklearn.cluster import DBSCAN, KMeans


__all__ = ['apply_clustering', 'apply_kmeans']


def apply_clustering(components_patient_info, eps = 1):
    
#     components = get_pacients_pca_components(dataset)
    
    components = [comp[2] for comp in components_patient_info]
    
    dbscan = DBSCAN(eps = eps).fit(components)
    
    return dbscan


def apply_kmeans(components_patient_info, clusters = 5):
    
    components = [comp[2] for comp in components_patient_info]
    
    kmeans = KMeans(n_clusters=clusters).fit(components)

    return kmeans