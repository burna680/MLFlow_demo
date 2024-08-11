#################################################################################
#-------------------------------Import libraries--------------------------------#
#################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#################################################################################
#------------------------------Imports dataset----------------------------------#
#################################################################################
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 

#################################################################################
#--------------------------Exploratory data analysis----------------------------#
#################################################################################
print(X.columns,y.columns)
print(X.info(), y.info())
print(X.head(), y.head())

#Frequency distribution of values in variables
for col in X.columns:
    print(X[col].value_counts())
print(y.value_counts())

# check missing values in variables
print(X.isnull().sum())


#################################################################################
#--------------------------------Data preparation-------------------------------#
#################################################################################
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.33, 
                                                    random_state = 42)

#################################################################################
#-------------------------------Feature engineering-----------------------------#
#################################################################################
print(X_train.dtypes, y_train.dtypes)

encoder = ce.OrdinalEncoder(cols= list(X.columns))
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


#################################################################################
#-------------------------------Model training----------------------------------#
#################################################################################

# create the classifier with n_estimators = 100
clf = RandomForestClassifier(n_estimators=100, random_state=0)
# fit the model to the training set
clf.fit(X_train, y_train)

#################################################################################
#-------------------------------Model evaluation--------------------------------#
#################################################################################

# Predict on the test set results
y_pred_100 = clf.predict(X_test)
# Check accuracy score
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))

# View the feature scores
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_scores) #the most important feature is safety and least important feature is doors


# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
# Add title to the graph
plt.title("Visualizing Important Features")

# Visualize the graph
plt.savefig('important_features.png')


#################################################################################
#-------------------------------Model re-training-------------------------------#
#################################################################################
# Retraining the model with only important features
X = X.drop(['doors'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.33,
                                                    random_state = 42)

encoder = ce.OrdinalEncoder(cols= list(X.columns))
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
clf = RandomForestClassifier(random_state=0)
# fit the model to the training set
clf.fit(X_train, y_train)
# Predict on the test set results
y_pred = clf.predict(X_test)
# Check accuracy score 
print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#################################################################################
#-------------------------------Results analysis--------------------------------#
#################################################################################
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print(classification_report(y_test, y_pred))