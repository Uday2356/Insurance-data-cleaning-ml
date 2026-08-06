import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore',category=FutureWarning) #to ignore the unwanted errors

data = pd.read_csv('insurance.csv')

df = pd.DataFrame(data)

v= df.shape#it tell the shape of data 
c = df.describe()#it tell the basic stats of the data
d =df.columns#it tell the name of columns
r = df.dtypes#it tell ne the data types of the column
e =df.head() #starting 5 
u =df.tail()# last 5 
y = df.info() #info of the data



t =df.isna().sum()#it tell the count of the null value in the dataset


numeric_columns = ['age' , 'bmi' , 'children' , 'charges']

# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[col] ,kde=True,bins = 20)

# plt.show()


fig ,ax = plt.subplots(2,2,figsize=(10,10))

sns.histplot(df['age'],bins=20 , kde = True , ax = ax[0,0])


sns.histplot(df['bmi'] , bins = 20 , kde = True , ax=ax[0,1])


sns.histplot(df['children'],bins = 20 , kde = True , ax = ax[1,0])


sns.histplot(df['charges'],bins = 20 , kde = True , ax = ax[1,1])
sns.countplot(x=df['children'])
sns.countplot(x=df['sex'])
sns.countplot(x=df['smoker'])
plt.close(True)

# plt.savefig('Numerical Data Comparison')
# plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.savefig("Correlations")
# plt.show()
plt.close(True)


