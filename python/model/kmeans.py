from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


class KMeansProcessor:
    def __init__(self, df_pop_pca, df_cust_pca):
        """
        Parameters
        ----------
            df_pop_pca : pd.DataFrame
                population data after PCA transformation
                
            df_cust_pca : pd.DataFrame
                custeomr data after PCA transformation
        """        
        self.model = None
        self.pop_labels = None
        self.pop_centroids = None
        self.cust_labels = None
        self.cust_centroids = None
        self.n_clsuters=None
        
        self.df_pop_pca = df_pop_pca
        self.df_cust_pca = df_cust_pca
        
        
    def fit(self, n_clusters):
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
        self.n_clusters = n_clusters
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=50000, random_state=3425)
        self.model = kmeans.fit(self.df_pop_pca)
        
    
    def predict(self):
        self.pop_labels = self.model.predict(self.df_pop_pca)
        self.pop_centroids = self.model.cluster_centers_
        
        self.cust_labels = self.model.predict(self.df_cust_pca)
        self.cust_centroids = self.model.cluster_centers_
                
    
    def get_cluster_proporations(self, return_as_pivot=True):
        """
        Description
        -----------
            calculates the proportion per cluster
        """        
        population_component_share = 100 * np.bincount(self.pop_labels)/len(self.pop_labels)
        customer_component_share = 100 * np.bincount(self.cust_labels)/len(self.cust_labels)
        component_shares = pd.DataFrame({'population':population_component_share,'customers':customer_component_share})
        component_shares = component_shares.reset_index().melt(value_vars=['population','customers'],id_vars=['index'])
        component_shares.rename(columns={'index':'cluster','variable':'dataset','value':'share'}, inplace=True)
        
        if return_as_pivot:
            component_shares = component_shares.pivot(columns='dataset', index='cluster')
            component_shares = component_shares.droplevel(0, axis=1)
            component_shares = component_shares.sort_values(by='customers', ascending=False)
            t = component_shares.cumsum()
            component_shares = pd.merge(left=component_shares, right=t, how='inner', left_index=True, right_index=True, suffixes=['_pct','_pctcum'])
            component_shares['diff_cumsum'] = component_shares['customers_pctcum'] - component_shares['population_pctcum']
            component_shares['diff'] = component_shares['customers_pct'] - component_shares['population_pct']
        
        return component_shares
        
    def plot_segmentation_comparison(self, ax):
        """
        Description
        ------------
            plots segmentation for customer and general population dataset for given KMeans Models

        Parameters
        -----------
            ax : mathplotlib.axes
                axes object that the data should be plotted to
        """


        # Compare the proportion of data in each cluster for the customer data to the
        # proportion of data in each cluster for the general population.
        population_component_share = 100 * np.bincount(self.pop_labels)/len(self.pop_labels)
        customer_component_share = 100 * np.bincount(self.cust_labels)/len(self.cust_labels)
        component_shares = pd.DataFrame({'population':population_component_share,'customers':customer_component_share})
        component_shares = component_shares.reset_index().melt(value_vars=['population','customers'],id_vars=['index'])
        component_shares.rename(columns={'index':'component','variable':'dataset','value':'share'}, inplace=True)
        
        sns.barplot(data=component_shares, x='component',y='share',hue='dataset',   ax=ax)
        
        ax.set_ylabel('Share in percent')
        ax.set_xlabel('Cluster')
        ax.set_xticks(range(0,self.model.n_clusters))
        ax.set_title('Customers vs Population Clusters')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', rotation=70)


    def show_component_details(self, pca_model, columns, sort_by_customer:bool=True, n_disp_max_clusters=3,n_disp_max_features=3):
        n_disp_max_clusters = n_disp_max_clusters
        n_disp_max_features = n_disp_max_features
        investigate_col_set_pos = set()
        investigate_col_set_neg = set()
    
    
        customer_centroids = self.model.cluster_centers_
        centroids_df = pd.DataFrame(pca_model.pca.inverse_transform(customer_centroids), columns=columns)

        if sort_by_customer:
            sort_by = 'customers_pct'
        else:
            sort_by = 'population_pct'
            
        proportions = self.get_cluster_proporations().sort_values(by=sort_by, ascending=False)



        for component_idx in proportions.index[:n_disp_max_clusters]:
            #print('-'*50)

            display(proportions.loc[[component_idx]])



            max_pos_cols = centroids_df.iloc[[component_idx],:].T.sort_values(by=component_idx, ascending=False)[:n_disp_max_features]
            for e in max_pos_cols.index:
                investigate_col_set_pos.add(e)

            max_neg_cols = centroids_df.iloc[[component_idx],:].T.sort_values(by=component_idx, ascending=False)[-n_disp_max_features:]
            for e in max_neg_cols.index:
                investigate_col_set_neg.add(e)                    

            print('Positive')
            display(max_pos_cols)        
            print()
            print('Negative')
            display(max_neg_cols)
            print()
            print('-'*50)
            print()

        return investigate_col_set_pos, investigate_col_set_neg
        
