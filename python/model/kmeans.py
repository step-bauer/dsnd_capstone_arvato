from sklearn.cluster import MiniBatchKMeans, KMeans

class KMeansProcessor:
    def __init__(self):
        self.model = None
        self.pop_labels = None
        self.pop_centroids = None
        
    def fit_transform(self, df_pca, n_clusters):
        """
        DESCRIPTION:
            Apply K-Means clustering to the dataset with a given number of clusters.

        INPUT:
            df_pca: dataset (usually with latent features)
            n_clusters: number of clusters to apply K-Means

        OUTPUT:
            gen_labels: labels (cluster no) for each data point
            gen_centroids: the list of coordinate of each centroid
            k_model (sklearn.cluster.k_means_.MiniBatchKMeans ): the cluster model 
        """

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=50000, random_state=3425)
        self.model = kmeans.fit(df_pca)
        self.pop_labels = self.model.predict(df_pca)
        self.pop_centroids = self.model.cluster_centers_

        return self.pop_labels, self.pop_centroids, self.model