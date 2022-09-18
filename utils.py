#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline ,FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import StandardScaler ,OneHotEncoder


# In[2]:


#file path to read it
FILE_PATH='./housing.csv'
# read file by pandas 
df=pd.read_csv(FILE_PATH)
#replace '<1H OCEAN' to 'OCEAN'
df['ocean_proximity']=df['ocean_proximity'].replace('<1H OCEAN','OCEAN')
#add more columns may be useful for model
df['room_per_houseshold']=df['total_rooms']/df['households']
df['bedromms_per_rooms']=df['total_bedrooms']/df['total_rooms']
df['population_per_houseshold']=df['population']/df['households']
X=df.drop(columns='median_house_value')
Y=df['median_house_value']
#split data to train ,test
X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True ,test_size=0.2,random_state=42)
num_cols=[col for col in X_train.columns if X_train[col].dtype in ['int32','int64','float32','float64'] ]
cat_cols=list(set(X_train.columns)-set(num_cols))


# In[3]:


num_pipeline=Pipeline(steps=[
    ('selector',DataFrameSelector(num_cols)),
    ('imputer',SimpleImputer(strategy='median')),
    ('scalar',StandardScaler())
])
pipeline_cat=Pipeline(steps=[
    ('selector',DataFrameSelector(cat_cols)),
    ('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
    ('ohe',OneHotEncoder(sparse=False))
])
total_pipeline=FeatureUnion(transformer_list=[
    ('num',num_pipeline),
    ('cat',pipeline_cat)
])
X_train_final=total_pipeline.fit_transform(X_train)


# In[4]:


def process_new(X_new):
    return total_pipeline.transform(X_new)


# In[ ]:




