import pandas as pd 
import numpy as np 
import statsmodels.api as sm 

df = pd.read_csv("fraud_dataset.csv")

# There are two cols that need to be changed to dummy variables
# Replace each of the current cols to dummy version. 
# Use 1 for weekday and True
# Use 0 for weenkend and False

# Way 1:
df[["not_fraud", "fraud"]] = pd.get_dummies(df["fraud"])
df.drop('not_fraud', axis=1, inplace=True)
df[["weekend", "weekday"]] = pd.get_dummies(df["day"])
df.drop("weekday", axis=1, inplace=True)
print (df.head(20))

# # Way 2:
# df["fraud"] = pd.get_dummies(df["fraud"])[True]
# df["day"] = pd.get_dummies(df["day"])["weekday"]
# print(df.head(20))