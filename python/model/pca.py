import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from python.etl.plot import Plot


class PCAProcessor:
    def __init__(self, X_train, metadata, pca_components=None, random_state=None):
        self.X_train = X_train
        self.X_train_standardized = None
        self.X_train_transformed = None
        self.metadata= metadata
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.scaler  = StandardScaler()
        self.pca_components = pca_components
        
        if pca_components:
            self.pca     = PCA(n_components=pca_components, random_state=random_state)
        else:
            self.pca     = PCA()

        
    def __scale_and_impute(self, X_train):
        """
        Description
        -----------
            this method scales data to [0,1] and imputes missing values
        
        
         Return
        ------
            pd.DataFrame : result scale and impute method
        """
        col_names = X_train.columns
        index = X_train.index
    
        # scale
        df = self.scaler.fit_transform(X_train)
        # impute
        df = pd.DataFrame(self.imputer.fit_transform(df))
        
        df.set_index(index, inplace=True)
        df.columns = col_names
        
        return df
                
        
            
    def fit_transform (self):      
        """
        Description
        -----------
        fits and transforms the data.
        
        * scales date and imputes missing data
        * runs pca fit_and transform
        * saves the result in X_train_transformed
        
        
        Return
        ------
            pd.DataFrame : result of fit and transform applied on self.X_train
        """
        index = self.X_train.index
        
        self.X_train_standardized = self.__scale_and_impute(self.X_train)
        self.X_train_transformed = self.pca.fit_transform(self.X_train_standardized)
        
        self.X_train_transformed = pd.DataFrame(self.X_train_transformed, index= index)
        
        return self.X_train_transformed
        
        
    def transform (self, df):
        col_names = self.X_train_transformed.columns
        index = self.X_train_transformed.index
        
        
        df = self.__scale_and_impute(df)
        df = self.pca.transform(df)
                        
        return df
    
        
        
        
 
    def plot (self, var_thresold):
        """
        Description
        -----------
            plots the PCA explained variance over number of components
            
        Parameters
        ----------
            var_threshold : float
                value between 0 and 1 that defines the percentage of explained variance. Used to draw a vertial line
        """
        n_component = self.get_components_for_variance(var_thresold)
        
        Plot.plot_exp_var_ratio(self.pca, hline_y=var_thresold, vline_x=n_component)
        
    def get_components_for_variance (self, x):
        """
        calculates the number of required components to explain x percent of the variance
        """
        cumvals = np.cumsum(self.pca.explained_variance_ratio_)
        idx = np.argmax(cumvals>x)
        
        return idx
        
    def get_component_features(self, n_component):
        """
        Description
        -----------
            Lists for a given component the feature ordered by importance
            
        Parameters
        ----------
            n_component : int
                component index
            
           
        """
        print(f'Explained Variance in % for cmponent: {n_component}: {self.pca.explained_variance_ratio_[n_component]:5.4f}')
        df_col_desc = self.metadata[['Attribute','Description']].drop_duplicates()
        df_comp = pd.DataFrame(self.pca.components_, columns=self.X_train.columns).iloc[n_component].sort_values(ascending=False)
        df_comp = df_comp.reset_index()
        df_comp.rename(columns={'index':'Attribute', n_component:'Variance'}, inplace=True)        
        
        return pd.merge(left=df_comp, right=df_col_desc, on='Attribute')
    