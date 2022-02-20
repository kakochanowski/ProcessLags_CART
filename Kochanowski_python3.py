"""
BIG DATA ASSIGNMENT 3
Identify bottlenecks in building permit processing times. Find main determinants of longer processing times measured
as the difference between the filing and the issue date.

Data from 1/1/2013 to 2/25/2018, includes 41 features, 200,000 observations

1. Find best predictors using
- Regression tree
- Regularized OLS regression
- One additional classifier
"""

# Import packages
import pandas as pd
import numpy as np
import seaborn as sbn; sbn.set()
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sbn; sbn.set()

"""PART ONE"""
# Read in data
data_original = pd.read_csv('assignment3_Building_Permits.csv')

# Create label variable (difference between filing and issue date)
data_original['Issue Date'] = pd.to_datetime(data_original['Issued Date'])
data_original['File Date'] = pd.to_datetime(data_original['Filed Date'])
data_original['Process_time'] = (data_original['Issue Date'] - data_original['File Date']).astype('timedelta64[D]')
# Change to date format for all other time variables
data_original['Permit Creation Date'] = pd.to_datetime(data_original['Permit Creation Date'])
data_original['Current Status Date'] = pd.to_datetime(data_original['Current Status Date'])
data_original['First Construction Document Date'] = pd.to_datetime(data_original['First Construction Document Date'])
data_original['Permit Expiration Date'] = pd.to_datetime(data_original['Permit Expiration Date'])
data_original['Completed Date'] = pd.to_datetime(data_original['Completed Date'])
# Create Year Filed variable
data_original['Year Filed'] = pd.DatetimeIndex(data_original['Year Filed']).year
# Create Permit Length variable
data_original['Permit Length'] = (data_original['Permit Expiration Date'] - data_original['Issue Date']).astype('timedelta64[D]')

# Determine variable types
print(data_original.dtypes)

# Create integers
data_original['Number of Existing Stories'] = round(data_original['Number of Existing Stories'])
data_original['Estimated Cost'] = round(data_original['Estimated Cost'])

# Make Fire Only Permit binary
data_original['Fire Only Permit_B'] = [1 if x == 'Y' else 0 for x in data_original['Fire Only Permit']]
data_original = data_original.drop('Fire Only Permit', axis=1)

# Bolster Voluntary Soft-Story Retrofit reporting given inconsistencies in data (based on Description var)
data_original['VSSR'] = data_original.Description.str.extract\
    ('(soft story|voluntary seismic upgrade|mandatory|manatory|mandated)') #accounting for inconsistent data logging
data_original['VSSR'] = data_original['VSSR'].fillna(0)
data_original.loc[data_original['VSSR'] == 0, 'Voluntary Soft-Story Retrofit V2'] = 0
data_original.loc[data_original['VSSR'] == 'mandatory', 'Voluntary Soft-Story Retrofit V2'] = 0
data_original.loc[data_original['VSSR'] == 'manatory', 'Voluntary Soft-Story Retrofit V2'] = 0
data_original.loc[data_original['VSSR'] == 'mandated', 'Voluntary Soft-Story Retrofit V2'] = 0
data_original.loc[data_original['VSSR'] == 'voluntary seismic upgrade', 'Voluntary Soft-Story Retrofit V2'] = 1
data_original.loc[data_original['VSSR'] == 'soft story', 'Voluntary Soft-Story Retrofit V2'] = 1
data_original = data_original.drop('VSSR', axis=1)
data_original = data_original.drop('Voluntary Soft-Story Retrofit', axis=1)

# Make 'Use Change' variable
data_original['Existing Use'] = data_original['Existing Use'].fillna('No Record')
data_original['Proposed Use'] = data_original['Proposed Use'].fillna('No Record')
data_original['Use Change'] = np.where(data_original['Existing Use'] == data_original['Proposed Use'], '0', '1')

# Make 'Units Change' variable
data_original['Existing Units'] = data_original['Existing Units'].fillna('No Record')
data_original['Proposed Units'] = data_original['Proposed Units'].fillna('No Record')
data_original['Units Change'] = np.where(data_original['Existing Units'] == data_original['Proposed Units'], '0', '1')

# Make 'Stories Change' variable
data_original['Stories Change'] = np.where(data_original['Number of Existing Stories'] ==
                                           data_original['Number of Proposed Stories'], '0', '1')

"""
Review variables:
- Average processing time is 26 days, median is zero days
- Removing some variables based off of intuition: 
    - street number suffix
    - street name
    - location (not enough variation, redundant with zipcode)
    - record id
    - permit number (sequentially assigned value)
    - permit type definition
    - lot (enough variation with block)
    - street number
    - unit
    - unit suffix (to avoid overfitting)
    - description
    - structural notification
    - estimated cost (revised cost very similar and less NAs)
    - completed date (not a useful predictor considering it follows the issued permit)
    - permit creation date
    - TIDF compliance (insignificant amount of information - can't be parsed with description info)
    - Structural notification (none with existent label)
    - Original formatting of Issued and Filed dates
- Dropping NAs from File Date variable (supervised model)
- Excluding some observations based off of Current Status variable - suspend, cancelled, expired
 (not valuable to our prediction/understanding of main determinants)
"""

# Summary stats
print(data_original.describe())
print(data_original['Process_time'].mean())
print(data_original['Process_time'].median())
print(data_original['Process_time'].min())
print(data_original['Process_time'].max())

# Drop all NAs for Issued Date variable (supervised model)
data_v1 = data_original.dropna(subset=['Issued Date'])

data_v1 = data_v1.dropna(subset=['Permit Expiration Date'])

# Drop irrelevant variables
data_v1 = data_v1.drop(['Issued Date', 'Filed Date', 'Permit Creation Date', 'Current Status Date',
                        'First Construction Document Date', 'Street Number Suffix', 'Street Name', 'Location',
                        'Record ID', 'Permit Number', 'Permit Type Definition', 'Lot', 'Street Number', 'Unit',
                        'Unit Suffix', 'Existing Construction Type Description', 'Proposed Construction Type Description', 'Estimated Cost', 'Completed Date', 'TIDF Compliance', 'Description',
                        'Structural Notification'], axis=1)
# Save as csv
data_v1.to_csv('data_v1.csv', index=False)

# Excluding observations based on 'Current Status'


# Date variables
data_dates = data_v1[['Process_time', 'File Date', 'Issue Date', 'Permit Expiration Date']]
# data_dates = pd.DataFrame(data=data_d, columns=['Process_time', 'File Date', 'Issue Date', 'Permit Expiration Date'])

# Replace NAs with feature average
data_v2 = data_v1.drop(['Issue Date', 'File Date', 'Permit Expiration Date', 'Process_time'], axis=1)
column_means = (data_v2.mean())
column_means = column_means.astype(float).round()
data_v3 = data_v2.fillna(column_means)
data_final = pd.concat([data_dates, data_v3], axis=1)

# Cleaning up some variables
data_final.loc[data_final['Proposed Units'] == 'No Record','Proposed Units'] = np.nan
data_final.loc[data_final['Proposed Use'] == 'No Record','Proposed Use'] = np.nan
data_final.loc[data_final['Existing Units'] == 'No Record','Existing Units'] = np.nan

# Save as csv
data_final.to_csv('data_final.csv', index=False)

"""PART TWO"""
#split cleaned data into test and train
# X_train, X_test, y_train, y_test = train_test_split(
       # cancer.data, cancer.target, stratify=cancer.target, random_state=42)

#split cleaned data into test and train
msk = np.random.rand(len(data_final)) < 0.80
data_final_train = data_final[msk]
data_final_test = data_final[~msk]
# Training data
xtrain = data_final_train.drop('Process_time', axis=1)
ytrain = data_final_train.loc[:, 'Process_time']
# Test data
xtest = data_final_test.drop('Process_time', axis=1)
ytest = data_final_test.loc[:, 'Process_time']

# look at some simple statistics for the survival rate of each feature
fn = list(xtrain.columns)
for feature in fn:
    print(data_final_train[[feature, 'Process_time']].groupby([feature], as_index=False).agg(['mean', 'count', 'sum']))

# Tree Model
tree = DecisionTreeClassifier(random_state=0)
tree.fit(xtrain, ytrain)

# compute accuracy in the test data
print("Tree's accuracy on test set: {:.3f}".format(tree.score(xtest, ytest)))
# cn = ['Died', 'Survived']
# plot_tree(tree, feature_names = fn, class_names = cn, filled = True)
# plt.show()



