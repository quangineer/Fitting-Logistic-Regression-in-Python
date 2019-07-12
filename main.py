import pandas as pd 
import numpy as np 
import statsmodels.api as sm 

df = pd.read_csv("fraud_dataset.csv")
# print (df.head(10))

# 1st, check dataset to see if any missing:
# print (df.describe(), df.info())

# Similiar to linear regression model, we want to change categorical 
# variables (non-numerical) into dummies (0,1)

df[["no_fraud", "fraud"]] = pd.get_dummies(df["fraud"])


# drop no_fraud column because of innecessary 
df = df.drop("no_fraud", axis = 1)
print (df.head())