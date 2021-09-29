"""
    Description
    -----------
        Utilities for the project
"""
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd

class Utils ():
    @classmethod
    def run_validation (cls, clf, X_val, y_val):
        y_pred = clf.predict(X_val)
        print(f'Classifier: {clf}')
        print(f'ROCAUC score: {roc_auc_score(y_val, y_pred)}')        
        

        df_confusion_matrix = pd.DataFrame(confusion_matrix(y_val, y_pred), columns=['Pred_False','Pred_True'], index=['Act_False','Act_True'])
        display(df_confusion_matrix)
    
    @classmethod
    def test_classifier (cls, clf, X_train, X_val, y_train, y_val):  
        """
        Description
        -----------
            helper funciton to test a classifier. runs fit and prints ROCAUC score and the confusion matrix
            
        Parameters
        ----------            
            clf : estimator
                fitted estimator
                
            X_train, X_val, y_train, y_val : pd.Dataframe
                train, validation data set and their labels
        """

        clf.fit(X_train, y_train)
        cls.run_validation(clf, X_val, y_val)


        return clf

    
    @classmethod
    def build_kaggle_file(cls, df, clf, filename):
        """
        Description
        -----------
            builds prediction file for Kaggle submission
            
        Parameters
        ----------
            df : pd.DataFrame
                test data set
            
            clf : estimator
                fitted estimator
                
            filename : str
                fielname (local fileformat or AWS s3 format)
        """
        y_predict = clf.predict_proba(df)
        pd.DataFrame(index=df.index, data=y_predict[:, 1], columns=['RESPONSE']).to_csv(filename)
        
        
        print(f'prediction file created: {filename}')