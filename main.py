import pandas as pd
import sklearn.model_selection as sk
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib  import dump, load

df=pd.read_csv("data.csv")
print(df.head())
print(df.info())
print(df["CHAS"].value_counts())
print(df.describe())

#Train and Test splitting
train_set,test_set=sk.train_test_split(df,test_size=0.2,random_state=42)
print(test_set["CHAS"].value_counts())
print(train_set["CHAS"].value_counts())

#stratifiedShuffleSplit (Equal amount of 0 and equal amount of 1 from CHAS column)
splitting = sk.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitting.split(df, df['CHAS']):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
print(strat_test_set['CHAS'].value_counts())
print(strat_train_set['CHAS'].value_counts())
#finding correlations
corr_matrix=df.corr()
relation=corr_matrix["MEDV"].sort_values(ascending=False)
print(relation)
# Name: count, dtype: int64
# MEDV       1.000000
# RM         0.695360
# ZN         0.360445
# B          0.333461
# DIS        0.249929
# CHAS       0.175260         no relation esy hta b skty agr values bht missing hy kuky final outcome k sath eska relation almost zero hy 
# AGE       -0.376955
# RAD       -0.381626
# CRIM      -0.388305
# NOX       -0.427321
# TAX       -0.468536
# INDUS     -0.483725
# PTRATIO   -0.507787
# LSTAT     -0.737663
# Name: MEDV, dtype: float64

#handling missing values
#option1: Get rid of all rows that contain missing values
#If a row has a NaN in even one column, that row is dropped.
a=df.dropna()

#option2: Get rid of Column that contain null values if it has no correlation
a=df.drop("RM",axis=1)   #means drop column name RM

#option3: fill null values with 0, mean and median
med=df["RM"].median
df["RM"].fillna(med)

# ********* Direct apply nai hogi csv file main kuky implace=True ni likha*************

#simplest way to do option3 is with  imputer
med =SimpleImputer(strategy="median")
med.fit(strat_train_set)
x=med.transform(strat_train_set)
df_transform=pd.DataFrame(x,columns=df.columns)
print(df_transform)
# fit() computes the statistics (e.g., median) on the dataset.
# transform() returns a new NumPy array with the missing values filled in.
# The original dataset (strat_train_set) remains unchanged.

#Seperating Feature and Labels 
housing_feature=strat_train_set.drop("MEDV",axis=1)
df_label=strat_train_set["MEDV"].copy()

#Creating pipeline
my_pipeline=Pipeline(
        [
                ("imputer", SimpleImputer(strategy="median")),
                ("std_scalar", StandardScaler())
        ]
)

housing_data_tranformed=my_pipeline.fit_transform(housing_feature)

#Selecting Desired model
# model=LinearRegression()
model=DecisionTreeRegressor()
model.fit(housing_data_tranformed,df_label)
# Select a Few Samples
some_data = housing_feature.iloc[:5]  # Use raw DataFrame with column names
some_labels = df_label.iloc[:5]
# transform(some_data) is used on new/unseen/test data.
# It uses the same median and scaling values learned from the training set
prepared_data = my_pipeline.transform(some_data)

print(model.predict(prepared_data))

# Evaluating metrices
housing_predictions = model.predict(housing_data_tranformed)
lin_mse = mean_squared_error(df_label, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_mse)

#cross validation
scores = cross_val_score(model, housing_data_tranformed, df_label, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)
print("rmse_scores",rmse_scores)
# Save trained model
dump(model, "real_estate_model.joblib")

# ave preprocessing pipeline
dump(my_pipeline, "real_estate_pipeline.joblib")

# *******TESTING**************
# Separate features and labels from test set
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
# Transform the test features using the same pipeline used for training
X_test_prepared = my_pipeline.transform(X_test)
print("Checking",X_test)
# Predict using the trained model

final_predictions = model.predict(X_test_prepared)
# Calculate the Mean Squared Error between predictions and actual values
final_mse = mean_squared_error(Y_test, final_predictions)

# Take the square root of MSE to get Root Mean Squared Error (RMSE)
final_rmse = np.sqrt(final_mse)
# Show final RMSE
print(final_rmse)
#comparing 
print("comparision")
print(final_predictions,list(Y_test))

print(X_test_prepared[0])












