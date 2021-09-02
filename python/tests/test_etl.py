import pandas as pd
import numpy as np
import ptvsd

import sys
import os
print (os.getcwd())
sys.path.append('.\\python')
print(sys.path)

import etl.processor as etlp

DEBUGGING=True

if DEBUGGING:
    ptvsd.enable_attach(address=('localhost', 5679))
    print('waiting for debugger....')
    ptvsd.wait_for_attach()

import os
if os.path.exists('data') and os.path.isdir('data'):
    prefix = './data'
else:
    prefix = 's3://sagemaker-eu-central-1-292575554790/dsnd_arvato'

#df_azdias = pd.read_csv(f'{prefix}/Udacity_AZDIAS_052018.csv', sep=';', index_col='LNR')
#df_customers = pd.read_csv(f'{prefix}/Udacity_CUSTOMERS_052018.csv', sep=';', index_col='LNR')
#df_metadata = pd.read_excel(f'{prefix}/DIAS Attributes - Values 2017.xlsx', usecols='B:E', dtype='str', header=1).fillna(method='ffill')

df_azdias = pd.read_csv(f'{prefix}/TestDebug_AZDIAS.csv', sep=';', index_col='LNR')
df_customers = pd.read_csv(f'{prefix}/TestDebug_CUSTOMERS.csv', sep=';', index_col='LNR')
df_metadata = pd.read_excel(f'{prefix}/DIAS Attributes - Values 2017.xlsx', usecols='B:E', dtype='str', header=1).fillna(method='ffill')


TESTING = False
if TESTING:
    df_azdias_cleaned = df_azdias.iloc[:100,:].copy()
else:
    df_azdias_cleaned = df_azdias.copy()

dfCleaner = etlp.PreDataCleaner(df_metadata)
df_azdias_cleaned = dfCleaner.transform(df_azdias_cleaned)

featureBuilder = etlp.FeatureBuilder()
df_azdias_cleaned = featureBuilder.transform(df_azdias_cleaned)
print(df_azdias_cleaned.shape)

print(df_azdias_cleaned['d_HAS_CHILDREN_YTE10'])
