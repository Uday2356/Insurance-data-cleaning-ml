import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from DC_DP import df_cleaned
from sklearn.preprocessing import  StandardScaler

#Feature engineering 



#we crerate bin for each category of the BMI as 

# [underweight , normal , overweight , obese]\

df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins = [0,18.5,24.9 , 29.9 ,float('inf')],
    labels= ['Underweight' , 'Normal' , 'Overweight' , 'Obese']

)

# 3 IT WILL BE IN STRING FROM SO WE OCNVERT IT INTO ONE HOT ENCODING 

df_cleaned = pd.get_dummies(df_cleaned,columns=['bmi_category'],drop_first=True)

df_cleaned['bmi_category_Normal'] = df_cleaned['bmi_category_Normal'].map({False:0,True:1})
df_cleaned['bmi_category_Overweight'] = df_cleaned['bmi_category_Overweight'].map({False:0,True:1})
df_cleaned['bmi_category_Obese'] = df_cleaned['bmi_category_Obese'].map({False:0,True:1})

df_cleaned = df_cleaned.astype(int) # this convert all float value  in integers

cols = ['age' , 'bmi' , 'children']

scal = StandardScaler()

df_cleaned[cols] = scal.fit_transform(df_cleaned[cols]) # this is feature scaling used for standardizations 
 #all values came between  -1 to 1



from scipy.stats import pearsonr

selected_features =['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges',
       'region_northwest', 'region_southeast', 'region_southwest',
       'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
       ]

Correlations = {
    features:pearsonr(df_cleaned[features],df_cleaned['charges'])[0] #it will find the corelation of charges with each member of selected _features
    for features in selected_features
}

#it return a dictionary 

correlations_df = pd.DataFrame(list(Correlations.items()) , columns=['features','Pearson Correlation'])


correlations_df.sort_values(by = 'Pearson Correlation' , ascending = False)





cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

from scipy.stats import chi2_contingency

alpha = 0.05

# Convert continuous charges into categorical bins
df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)

chi2_results = {}

for col in cat_features:
    # Create contingency table
    contingency = pd.crosstab(
        df_cleaned[col],
        df_cleaned['charges_bin']
    )

    # Chi-square test
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    # Decision based on p-value
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Fail to Reject Null (Drop Feature)'

    chi2_results[col] = {
        'Chi2 Statistic': chi2_stat,
        'P-value': p_val,
        'Decision': decision
    }

# Convert results into dataframe
chi2_results_df = pd.DataFrame(chi2_results).T


final_df = df_cleaned[
    [
        'is_smoker',
        'region_southeast',
        'is_female',
        'children',
        'age',
        'charges',
        'bmi',
        'bmi_category_Obese'
    ]
]
print(final_df)










