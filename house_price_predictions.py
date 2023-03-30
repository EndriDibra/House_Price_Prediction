# Author: Endri Dibra

# importing the required libraries for the project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the data from an excel file
df = pd.read_excel("house_price_prediction.xlsx")

# visualizing data using a heatmap (kind)
sns.heatmap(df.corr(), annot=True)

dimensions = df.shape
print("Dimensions of the dataset: " ,dimensions)

# showing the first 5 rows of the dataset
print(df.head())

print("Columns names: ", df.columns)

print(df.dtypes)

# checking some info about the dataset
print(df.info())

# more info about the values of the dataset
print(df.describe())

# removing some columns from the dataset
df.drop(['Id'], axis=1, inplace=True)
df = df.drop(["MSZoning"], axis=1)
df = df.drop(["LotConfig"], axis=1)
df = df.drop(["Exterior1st"], axis=1)
df = df.drop(["BldgType"], axis=1)

# checking for null values in dataset, if true then removing them
if df.isna().sum().any():

    df = df.dropna()

# checking for duplicates in dataset, if true then removing them
if df.duplicated().sum() > 0:

    df = df.drop_duplicates()

# splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

df['SalePrice'] = df['SalePrice'].fillna(df['SalePrice'].mean())

# training set
X = df.drop(["SalePrice"], axis=1)

# test set
Y = df["SalePrice"]

# 80% for training set and 20% for test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8,  test_size=0.2, random_state=0)

# using linear regression model from linear models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# creating the LM model
lm_model = LinearRegression()
lm_model.fit(X_train, Y_train)
Y_pred = lm_model.predict(X_test)

# checking for the performance of our LM model
print("Performance: ",mean_absolute_percentage_error(Y_test, Y_pred))

# storing the predictions
predictions = lm_model.predict(X_test)

# visualizing predictions
plt.scatter(Y_test,predictions)
plt.show()
