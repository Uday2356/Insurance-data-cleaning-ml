import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from EDA import df 

df_cleaned = df.copy() #this help to create a copy of whole dataset

t=df_cleaned.head()
y = df_cleaned.drop_duplicates(inplace=True)
d = df.shape

count = df['sex'].value_counts()
smoke = df['smoker'].value_counts()

#encoding of sex columns 
df_cleaned['sex'] = df_cleaned['sex'].map({'female':1,'male':1})

#encoding of Smoker column 
df_cleaned['smoker'] = df_cleaned['smoker'].map({'no':0,'yes':1})


#rename the column

df_cleaned.rename(columns={
    'sex' : 'is_female',
    'smoker':'is_smoker'
},inplace=True)

u = df_cleaned['region'].value_counts()


# attributes do not take () after them as we have shape , column they are attrubute 


#ONE HOT CODING 

df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)

# df_cleaned → DataFrame to modify.
# columns=['region'] → Convert the region column into dummy variables.
# drop_first=True → Drop the first category to avoid multicollinearity.
# The result is assigned back to df_cleaned because get_dummies() returns a new DataFrame; it does not modify the original one.



df_cleaned['region_northwest'] = df_cleaned['region_northwest'].map({False:0,True:1})
df_cleaned['region_southeast'] = df_cleaned['region_southeast'].map({False:0,True:1})
df_cleaned['region_southwest'] = df_cleaned['region_southwest'].map({False:0,True:1})
# print(df_cleaned.head())



