"""
Description
-----------

    This module is linked to the Arvato Project Workbook (jupyterlab notebook). Many explanation is given in the workbook. 


"""

class PreDataCleaner:
    """
    Description
    -----------

    This class will provide some functionality to execute some basic cleanining 
    a arvato data set
    """
    def __init__(self, df_metadata):
        """
        Description
        -----------

        inits the class.
        
        Parameters
        ----------
            df_metadata: pd.DataFrame
                pandas dataframe with the loaded data from file "DIAS Attributes - Values 2017.xlsx". Containing
                information about the attribute values.
        """
        self.df_metadata = df_metadata

        # replace the metadata attribute column ending "_RZ" by "" in order to match the dataset column names
        self.df_metadata['Attribute'] = self.df_metadata['Attribute'].str.replace('_RZ','')

    
    def transform(self, df):
        """
        Description
        -----------
                
        executes the data transformation (cleaning)

        Parameters
        ----------

            df : pd.DataFrame
                the dataframe that is to be cleaned

        """        
        df = self.__mark_nans(df)        
        df = self.__handle_data_load_errors(df)
        df = self.__catvars_to_dummies(df)        
        df = self.__catvars_to_binary(df)        
        df = self.__drop_columns(df)        
        return df
    
    def fit (self, df):
        """
        Description
        -----------

        prepare data for transformation
        """

        pass
    
    def __handle_data_load_errors(self, df):
        """
        handles the errors fo columns 18 and 19 of dtype float that contain two 18,19 
        """
        cols_to_fix = {'CAMEO_DEUG_2015':'X', 'CAMEO_INTL_2015':'XX'}

        print(f'fixing load errors {cols_to_fix}')

        for col, val in cols_to_fix.items():
            n = df.loc[df[col] == val].shape[0]
            df.loc[df[col] == val, col] = np.NaN
            df.loc[:,col] = df.loc[:,col].astype('float')

            print(f'fixed column {col} - records fixed: {n}')
        
        return df


    def __drop_columns(self, df, columns_to_drop=None):
        """
        Description
        -----------

        """
        # if columns to drop have been defined then use them 
        # else execute the default cleaning
        if columns_to_drop:            
            cols_to_drop = columns_to_drop
        else:
            cols_to_drop = ['EINGEFUEGT_AM'] 

        print(f'dropping columns: {cols_to_drop}')                

        try:
            df.drop(labels=cols_to_drop, axis=1, inplace=True)   
        except KeyError as ex_keyerror:
            print(f'CATCHED EXCEPTION: KeyError: you tried to drop non existing columns: {cols_to_drop}')
            print(f'Failed columns: {ex_keyerror.args}')

        return df     

    def __catvars_to_dummies(self, df):
        """
        Description
        -----------

        handles categorical variables. This will generate one hot encodings for the defined columns
        """
        cat_cols = ['CAMEO_DEU_2015','D19_LETZTER_KAUF_BRANCHE']

        print('creating one hot encoding columns for: ')
        for col in cat_cols:
            print(f'\t{col}')

        # create one hot encodings using pandas get_dummies function
        df_dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=True).astype('int64')
        df = pd.concat([df, df_dummies], axis=1)
        
        # drop original columns
        df.drop(cat_cols, axis=1, inplace=True)

        return df

    def __catvars_to_binary(self, df):
        """
        Description
        -----------

        """
        cat_cols = {'OST_WEST_KZ':{'W':0,'O':1}}

        print('convert to binary: ')
        for col, dict_map in cat_cols.items():
            print(f'\tcolumn: {col} - Mapping: {dict_map}')
            df.loc[:,col] = df.loc[:,col].map(dict_map)

        return df



    def __mark_nans(self, df):
        """
        Description
        -----------

        replaces all unkown values by np.NAN so that the pandas NAN functions can be used.

        Parameters
        ----------

            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                
        """       

        print('replace unkown values by NaNs: ') 
        unknown_val_set = df_metadata.copy()
        unknown_val_set = unknown_val_set[unknown_val_set['Meaning'].str.contains('unknown')]
        unknown_val_set['value_list']  = unknown_val_set['Value'].str.split(',')
        
        #with progressbar.ProgressBar(max_value=unknown_val_set.index.shape[0]) as bar:
        cnt = 0
        max_value=unknown_val_set.index.shape[0]
        for idx in unknown_val_set.index:
            col  = unknown_val_set.loc[idx,'Attribute']
            vals = unknown_val_set.loc[idx,'value_list']
            # str convert to integers
            vals = list(map(int,vals))
            if col in df:
                df.loc[df[col].isin(vals),col] = np.NaN

            cnt += 1
            if (cnt == max_value) or (cnt % (max_value // 10)==0):
                print(f'\tProcessed columns\r{cnt:4} of {max_value}', end='')
        
        print()
        return df
                                    
    @property
    def df_metadata(self):
        return self.__df_metadata
    
    @df_metadata.setter
    def df_metadata(self, val):
        self.__df_metadata = val
        

class FeatureBuilder:
    """
    Description
    -----------

    executes some data transformations on a arvato dataset and creates some new features
    """
    
    def __init__(self):
        pass

    def transform(self, df):
        """
        Description
        -----------

        executes the data transformation 

        Parameters
        ----------
            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                


        """
        self.__build_features_chidren(df)

        return df
    
    def fit (self, df):
        """
        Description
        -----------
        
        prepare data for transformation
        """
        pass

    def __build_features_chidren(self, df):
        """
        Description
        -----------
        
        This function will build some features based on the given input data

        * Children and Teens: 
            * Children:= number of children younger or equal than 10
            * Teens   := number of children older or equal than 10

        Parameters
        ----------
            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                
        """
        #df['d_NUM_CHILDREN_LESS_10'] = 0
        #df['d_NUM_CHILDREN_GTE_10'] = 0
        df['d_HAS_CHILDREN'] = 0
        df['d_HAS_CHILDREN_YTE10'] = 0
        
        cols = ['ANZ_KINDER','ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4',
                #'d_NUM_CHILDREN_LESS_10','d_NUM_CHILDREN_GTE_10',
                'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'
                ]

        #df.loc[df['ANZ_KINDER'] > 0,cols] = df.loc[df['ANZ_KINDER'] > 0,cols].apply(self.__calc_child_and_teens,'columns')
        df.loc[df['ANZ_KINDER'] > 0,cols] = df.loc[df['ANZ_KINDER'] > 0,cols].apply(self.__calc_children_features,'columns')
        df.drop('')

        return df
        


    def __calc_children_features(self, s):
        """
        Description
        -----------
            uses features 'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER' to reduce them to 
            'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'


            * d_HAS_CHILDREN_YTE10 if person has children ANZ_KINDER>0
            * d_HAS_CHILDREN if person has at least one children <= 10            

        Parameters
        ----------
            s : pd.Series
                series of a particular DataFrame row containing at least these columns
                'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER', 'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'
        """        
        yte_10 = (s[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']] <= 10).sum()
        

        s['d_HAS_CHILDREN'] = s['ANZ_KINDER']>0
        s['d_HAS_CHILDREN_YTE10']  = yte_10>0
        
        return s

    def __calc_child_and_teens(self, s):
        """
        Description
        -----------

        counts the number of children less 10 and greater equal than 10. I assume that for more than 5 children
        all children > 4 are older than 10. Based on the analysis this is in general true

        Parameters
        ----------
            s : pd.Series
                series of a particular DataFrame row containing at least these columns
                'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER', 'd_NUM_CHILDREN_LESS_10', 'd_NUM_CHILDREN_GTE_10'
        """        
        less_10 = (s[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']] < 10).sum()
        gte_10 = s['ANZ_KINDER'] - less_10

        s['d_NUM_CHILDREN_LESS_10'] = less_10
        s['d_NUM_CHILDREN_GTE_10']  = gte_10
        
        return s
    