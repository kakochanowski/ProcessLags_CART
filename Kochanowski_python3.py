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
# Feature engineering
import missingno as msno
# Data visualization
import seaborn as sbn; sbn.set()
import matplotlib.pyplot as plt
# Model testing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sbn; sbn.set()
# Regularized Regression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
# Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import memory_profiler

"""PART ONE"""
# Read in data
data_original_v1 = pd.read_csv('assignment3_Building_Permits.csv')
data_original_v1.describe()
# 198899 observations, 168221 unique permit numbers
# Drop all NAs for Issued Date variable (supervised model)
data_original_v2 = data_original_v1.dropna(subset=['Issued Date'])

# Determine level of observation
num_obs = data_original_v2.count
print(num_obs)
num_permits = data_original_v2['Permit Number'].value_counts()
num_ID = data_original_v2['Record ID'].value_counts()
print(num_permits)
print(num_ID)
# More observations than unique permit IDs. Drop observations with same permit ID given this is our variable of interest
data_original_v2 = data_original_v1.drop_duplicates('Permit Number', keep='last')

# Create label variable (difference between filing and issue date)
data_original_v2['Issue Date'] = pd.to_datetime(data_original_v2['Issued Date'])
data_original_v2['File Date'] = pd.to_datetime(data_original_v2['Filed Date'])
# Create different labels (days, weeks, more than one day)
# days
data_original_v2['Process_time_days'] = (data_original_v2['Issue Date'] - data_original_v2['File Date']).astype('timedelta64[D]')
avg_process = (data_original_v2['Process_time_days'].median())
print(avg_process)
# weeks
data_original_v2['Process_time_weeks'] = (data_original_v2['Process_time_days'] / 7)
data_original_v2['Process_time_weeks'] = round(data_original_v2['Process_time_weeks'])
# more than 1 week (given average process time is between 0 and 7 days, captures outliers well)
# Visualize distribution
bin_list = [0, 5, 15, 20, 25, 30, 35, 40, 45, 50]
x = np.array(data_original_v2['Process_time_days'])
plt.hist([x], bins=bin_list)
plt.title("Distribution of Process Time")
plt.savefig("process_time_hist.png")
plt.show()
data_original_v2['Process_time_type'] = [0 if x == 0 else 1 for x in data_original_v2['Process_time_weeks']]

# Change to date format for all other time variables
data_original_v2['Permit Creation Date'] = pd.to_datetime(data_original_v2['Permit Creation Date'])
data_original_v2['Current Status Date'] = pd.to_datetime(data_original_v2['Current Status Date'])
data_original_v2['First Construction Document Date'] = pd.to_datetime(data_original_v2['First Construction Document Date'])
data_original_v2['Permit Expiration Date'] = pd.to_datetime(data_original_v2['Permit Expiration Date'])
data_original_v2['Completed Date'] = pd.to_datetime(data_original_v2['Completed Date'])

# Determine variable types & null values
print(data_original_v2.dtypes)
data_original_v2.isnull().sum()/len(data_original_v2)

# Accounting for seasonality
# Create Year Filed variable
data_original_v2['Year Filed'] = pd.DatetimeIndex(data_original_v2['File Date']).year
# Create Quarter Filed variable
data_original_v2['Month Filed'] = pd.DatetimeIndex(data_original_v2['File Date']).month
data_original_v2.loc[data_original_v2['Month Filed'] <= 3, 'Quarter Filed'] = 1
data_original_v2.loc[(data_original_v2['Month Filed'] > 3) & (data_original_v2['Month Filed'] <= 6),
                  'Quarter Filed'] = 2
data_original_v2.loc[(data_original_v2['Month Filed'] > 6) & (data_original_v2['Month Filed'] <= 9),
                  'Quarter Filed'] = 3
data_original_v2.loc[(data_original_v2['Month Filed'] > 9) & (data_original_v2['Month Filed'] <= 12),
                  'Quarter Filed'] = 4
# Create weekday variable (discrete and only on weekdays)
data_original_v2['Day of Week'] = pd.DatetimeIndex(data_original_v2['File Date']).weekday

# Create Permit Length variable
data_original_v2['Permit Length'] = (data_original_v2['Permit Expiration Date'] - data_original_v2['Issue Date']).astype('timedelta64[D]')
data_original_v2['Permit Length'] = data_original_v2['Permit Length'].fillna(4000)
# Create Permit Duration variable (<6 mo, 1 yr, 2 yr, 5yr, 10 yr, indefinite)
data_original_v2.loc[data_original_v2['Permit Length'] <= 180, 'Permit Duration'] = 1
data_original_v2.loc[(data_original_v2['Permit Length'] > 180) & (data_original_v2['Permit Length'] <= 365),
                  'Permit Duration'] = 2
data_original_v2.loc[(data_original_v2['Permit Length'] > 365) & (data_original_v2['Permit Length'] <= 730),
                  'Permit Duration'] = 3
data_original_v2.loc[(data_original_v2['Permit Length'] > 730) & (data_original_v2['Permit Length'] <= 1825),
                  'Permit Duration'] = 4
data_original_v2.loc[(data_original_v2['Permit Length'] > 1825) & (data_original_v2['Permit Length'] <= 3650),
                  'Permit Duration'] = 5
data_original_v2.loc[data_original_v2['Permit Length'] > 3650, 'Permit Duration'] = 6
data_original_v2 = data_original_v2.drop(['Permit Length', 'Permit Expiration Date'], axis=1)

# Create integers (with NAs)
data_original_v2['Number of Existing Stories'] = round(data_original_v2['Number of Existing Stories'])
data_original_v2['Estimated Cost'] = round(data_original_v2['Estimated Cost'])

# Make Site Permit binary (lots of NAs)
data_original_v2['Site Permit Logged'] = [1 if x == 'Y' else 0 for x in data_original_v2['Site Permit']]
data_original_v2 = data_original_v2.drop('Site Permit', axis=1)

# Make Fire Only Permit binary (lots of NAs)
data_original_v2['Fire Only Permit_B'] = [1 if x == 'Y' else 0 for x in data_original_v2['Fire Only Permit']]
data_original_v2 = data_original_v2.drop('Fire Only Permit', axis=1)

# Bolster Voluntary Soft-Story Retrofit reporting given inconsistencies in data (based on Description var)
data_original_v2['VSSR'] = data_original_v2.Description.str.extract\
    ('(soft story|voluntary seismic upgrade|mandatory|manatory|mandated)') #accounting for inconsistent data logging
data_original_v2['VSSR'] = data_original_v2['VSSR'].fillna(0)
data_original_v2.loc[data_original_v2['VSSR'] == 0, 'Voluntary Soft-Story Retrofit V2'] = 0
data_original_v2.loc[data_original_v2['VSSR'] == 'mandatory', 'Voluntary Soft-Story Retrofit V2'] = 0
data_original_v2.loc[data_original_v2['VSSR'] == 'manatory', 'Voluntary Soft-Story Retrofit V2'] = 0
data_original_v2.loc[data_original_v2['VSSR'] == 'mandated', 'Voluntary Soft-Story Retrofit V2'] = 0
data_original_v2.loc[data_original_v2['VSSR'] == 'voluntary seismic upgrade', 'Voluntary Soft-Story Retrofit V2'] = 1
data_original_v2.loc[data_original_v2['VSSR'] == 'soft story', 'Voluntary Soft-Story Retrofit V2'] = 1
data_original_v2 = data_original_v2.drop('VSSR', axis=1)
data_original_v2 = data_original_v2.drop('Voluntary Soft-Story Retrofit', axis=1)

"""
Determine where NA values are still prevalent and whether some are duplications of information
"""
# Correlation
corr_matrix = data_original_v2.corr()
print(corr_matrix)

# Missing values
msno.matrix(data_original_v2)
data_original_v2.isnull().sum()/len(data_original_v2)

# Units with many missing values and correlated with permit type and use (information not collected). Will drop

# Improve Structural Notification variable given NAs (add "reported" var)
data_original_v2['Structural Notification Reported'] = 0
data_original_v2.loc[data_original_v2['Structural Notification'] == 'Y', 'Structural Notification Reported'] = 1

# Improve Existing Use n/a values
data_original_UseNoNA = data_original_v2[data_original_v2['Existing Use'].notnull()]
combinedUse = pd.concat([data_original_v2, data_original_UseNoNA])
data_original_UseNA = combinedUse.drop_duplicates(keep=False)
data_original_UseNA.loc[data_original_UseNA['Permit Type'] == 1, 'Existing Use'] = 'new'
data_original_UseNA.loc[data_original_UseNA['Permit Type'] == 2, 'Existing Use'] = 'new'
data_original_UseNA.loc[data_original_UseNA['Permit Type'] == 8, 'Existing Use'] = 'street space'
data_original_UseNA.loc[data_original_UseNA['Permit Type'] == 7, 'Existing Use'] = 'not applicable'
data_original_v2 = pd.concat([data_original_UseNoNA, data_original_UseNA])

# Improve Proposed Use n/a values
data_original_PropNoNA = data_original_v2[data_original_v2['Proposed Use'].notnull()]
combinedProp = pd.concat([data_original_v2, data_original_PropNoNA])
data_original_PropNA = combinedProp.drop_duplicates(keep=False)
data_original_PropNA.loc[data_original_PropNA['Permit Type'] == 1, 'Proposed Use'] = 'new'
data_original_PropNA.loc[data_original_PropNA['Permit Type'] == 2, 'Proposed Use'] = 'new'
data_original_PropNA.loc[data_original_PropNA['Permit Type'] == 8, 'Proposed Use'] = 'street space'
data_original_PropNA.loc[data_original_PropNA['Permit Type'] == 7, 'Existing Use'] = 'not applicable'
data_v1 = pd.concat([data_original_PropNoNA, data_original_PropNA])

# Make 'Use Change' variable
data_v1['Use Change'] = np.where(data_v1['Existing Use'] == data_v1['Proposed Use'], 0, 1)
data_v1.loc[data_v1['Permit Type'] == 6, 'Use Change'] = 1
data_v1.loc[data_v1['Existing Use'] == 'street space', 'Use Change'] = 0
data_v1['Use Change'] = data_v1['Use Change'].astype(int)

# Make 'Stories Change' variable
data_v1['Stories Change'] = np.where(data_v1['Number of Existing Stories'] ==
                                        data_v1['Number of Proposed Stories'], '0', '1')
# Stories is irrelevant for permit type 6 (demolition
data_v1.loc[data_v1['Permit Type'] == 6, 'Stories Change'] = 0

# Bin stories by building height (low-rise, mid-rise, high-rise)
# Existing
data_v1['Existing Build Rise'] = 0
data_v1.loc[data_v1['Number of Existing Stories'] <= 4, 'Existing Building Rise'] = 1
data_v1.loc[(data_v1['Number of Existing Stories'] > 4) & (data_v1['Number of Existing Stories'] <= 9),
                  'Existing Building Rise'] = 2
data_v1.loc[data_v1['Number of Existing Stories'] > 9, 'Existing Building Rise'] = 3
data_v1.loc[data_v1['Number of Existing Stories'] > 9, 'Existing Building Rise'] = 3
data_v1['Existing Building Rise'].replace(0, np.nan, inplace=True) # drop NA/s

# Proposed
data_v1['Proposed Building Rise'] = 0
data_v1.loc[data_v1['Number of Proposed Stories'] <= 4, 'Proposed Building Rise'] = 1
data_v1.loc[(data_v1['Number of Proposed Stories'] > 4) & (data_v1['Number of Proposed Stories'] <= 9),
                  'Proposed Building Rise'] = 2
data_v1.loc[data_v1['Number of Proposed Stories'] > 9, 'Proposed Building Rise'] = 3
data_v1['Proposed Building Rise'].replace(0, np.nan, inplace=True) # drop NA/s
# Dropping story variables (replaced by more generalized 'rise' variable)
data_v1 = data_v1.drop(['Number of Existing Stories', 'Number of Proposed Stories'], axis=1)
# Rise change
data_v1['Rise Change'] = np.where(data_v1['Existing Building Rise'] == data_v1['Proposed Building Rise'], '0', '1')

# Group Street Suffix (could be useful when understanding traffic constraints)
# 1 lightly trafficked areas, 2 medium, 3 high
data_v1['Street Traffic'] = 0
data_v1.loc[(data_v1['Street Suffix'] == 'Al') | (data_v1['Street Suffix'] == 'Cr') | (data_v1['Street Suffix'] == 'Ct')
                            | (data_v1['Street Suffix'] == 'Hl') | (data_v1['Street Suffix'] == 'Tr')
                            | (data_v1['Street Suffix'] == 'Ln'), 'Street Traffic'] = 1
data_v1.loc[(data_v1['Street Suffix'] == 'St') | (data_v1['Street Suffix'] == 'Pk') | (data_v1['Street Suffix'] == 'Pz')
                            | (data_v1['Street Suffix'] == 'Rd') | (data_v1['Street Suffix'] == 'Wy')
                            | (data_v1['Street Suffix'] == 'Dr'), 'Street Traffic'] = 2
data_v1.loc[(data_v1['Street Suffix'] == 'Av') | (data_v1['Street Suffix'] == 'Bl') | (data_v1['Street Suffix'] == 'Hy')
                            | (data_v1['Street Suffix'] == 'Rd') | (data_v1['Street Suffix'] == 'Wy')
                            | (data_v1['Street Suffix'] == 'Rw'), 'Street Traffic'] = 3
data_v1 = data_v1[data_v1['Street Traffic'] > 0] # drop NAs (previously assigned 0)

"""
Review variables:
- Average processing time is 26 days, median is zero days
- Removing some variables based off of intuition: 
    - Sequentially assigned values: record ID, permit number, street number suffix
    - Too much variation: location, lot, block, street number, unit suffix, street name
    - Duplicate information: description, location, permit type definition, estimated cost 
        (revised cost very similar and less NAs), strucutral notification (reported var instead), neighborhood (zipcode)
    - Insignificant: completed date (not a useful predictor considering it follows the issued permit), 
        permit creation date, first construction document date, TIDF compliance (insignificant amount of information - 
        can't be parsed with description info), original formatting of 
        Issued and Filed dates
- Dropping NAs from Issued Date variable (supervised model)
- Excluding some observations based off of Current Status variable - suspend, cancelled, expired
 (not valuable to our prediction/understanding of main determinants)
"""

# Summary stats
print(data_v1.describe())
print(data_v1['Process_time_days'].mean())
print(data_v1['Process_time_days'].median())
print(data_v1['Process_time_days'].mode())
print(data_v1['Process_time_days'].min())
print(data_v1['Process_time_days'].max())

# Save as csv
data_v1.to_csv('data_v1.csv', index=False)

# Drop irrelevant variables
data_v2 = data_v1.drop(['Issued Date', 'Filed Date', 'Issue Date', 'File Date', 'Permit Creation Date', 'Month Filed',
                        'Current Status Date', 'First Construction Document Date',
                        'Street Number Suffix', 'Street Suffix', 'Street Name', 'Location', 'Record ID', 'Permit Number',
                        'Permit Type Definition', 'Lot', 'Street Number', 'Unit', 'Unit Suffix',
                        'Existing Construction Type Description', 'Proposed Construction Type Description',
                        'Completed Date', 'TIDF Compliance', 'Description', 'Block',
                        'Structural Notification', 'Process_time_days', 'Process_time_type',
                        'Neighborhoods - Analysis Boundaries'], axis=1)


# Excluding observations based on 'Current Status'
data_v2.drop(data_v2[(data_v2['Current Status'] == 'cancelled') & (data_v2['Current Status'] == 'incomplete') &
                     (data_v2['Current Status'] == 'withdrawn')].index)
data_v2 = data_v2.drop(['Current Status'], axis=1)

# Determine how to deal with remaining missing values
sbn.heatmap(data_v2.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')

# Replacing NAs with median/feature given skewness
column_avgs = (data_v2.median())
column_avgs = column_avgs.astype(float).round()
data_final = data_v2.fillna(column_avgs)

# Save as csv
data_final.to_csv('data_final.csv', index=False)


"""
PART TWO: Prepare data for CART & regression
"""
# Determine variable types
print(data_final.dtypes)
num_na = data_final.isna().sum()
print(num_na)
data_final.to_csv('data_final.csv', index=False)
final = data_final.dropna()
# Save as csv
final.to_csv('final.csv', index=False)

"""
Visualize relationship between class and features
"""

# Plot each feature against quality
# data_final.columns = data_final.columns.astype(str)
# for i in data_final.columns:
   # sbn.boxplot(y=np.array(data_final["Process_time_weeks"]), x=np.array(data_final[i]))
   # plt.ylabel("Process time (weeks)")
   # plt.xlabel(str(i))
   # plt.title("Process time and " + i)
   # plt.savefig("figure"+"_"+str(i)+".png")


"""PART THREE"""
#split cleaned data into test and train
# Process time (weeks) produces an accuracy of ~65%. Going to use a binary label instead
msk = np.random.rand(len(final)) < 0.80
final_train = final[msk]
final_test = final[~msk]
# Training data
xtrain = final_train.drop('Process_time_weeks', axis=1)
ytrain = final_train.loc[:, 'Process_time_weeks']
# Test data
xtest = final_test.drop('Process_time_weeks', axis=1)
ytest = final_test.loc[:, 'Process_time_weeks']

# look at some simple statistics for process time of each feature
fn = list(xtrain.columns)
for feature in fn:
    print(final_train[[feature, 'Process_time_weeks']].groupby([feature], as_index=False).agg(['mean', 'count', 'sum']))

# Tree Model
tree = DecisionTreeClassifier(random_state=0)
tree.fit(xtrain, ytrain)

# compute accuracy in the test data
print("Tree's accuracy on test set: {:.3f}".format(tree.score(xtest, ytest)))
# plot_tree(tree, feature_names = fn, class_names = cn, filled = True)
# plt.show()
#


"""PRUNED TREE"""
# Reduce df to sample size so CCP can run
#split cleaned data into test and train
sample_tree = final.sample(n=1000)
sample_tree.shape
#split cleaned data into test and train
msk = np.random.rand(len(sample_tree)) < 0.80
final_train_s = sample_tree[msk]
final_test_s = sample_tree[~msk]
# Training data
xtrain_s = final_train_s.drop('Process_time_weeks', axis=1)
ytrain_s = final_train_s.loc[:, 'Process_time_weeks']
# Test data
xtest_s = final_test_s.drop('Process_time_weeks', axis=1)
ytest_s = final_test_s.loc[:, 'Process_time_weeks']

# Tree Model
tree_s = DecisionTreeClassifier(random_state=0)
tree_s.fit(xtrain_s, ytrain_s)

# compute accuracy in the test data
print("Sample Tree's accuracy on test set: {:.3f}".format(tree_s.score(xtest_s, ytest_s)))
# cn = ['Died', 'Survived']
# plot_tree(tree, feature_names = fn, class_names = cn, filled = True)
# plt.show()

# now find the best alpha value with which to prune
path = tree_s.cost_complexity_pruning_path(xtrain_s, ytrain_s)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    clf.fit(xtrain_s, ytrain_s)
    clfs.append(clf)
# drop the last model because that only has 1 node
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(xtrain_s)
    y_test_pred = c.predict(xtest_s)
    train_acc.append(accuracy_score(y_train_pred, ytrain_s))
    test_acc.append(accuracy_score(y_test_pred, ytest_s))

# plot the changes in accuracy for each alpha
plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, test_acc)
plt.plot(ccp_alphas, train_acc, label = 'train_accuracy', drawstyle = "steps-post")
plt.plot(ccp_alphas, test_acc, label = 'test_accuracy', drawstyle = "steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()
# the best outcome looks to be around 0.01

# the pruned tree model
# pruned_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=0.01)
# pruned_tree.fit(xtrain, ytrain)

# compute accuracy in the test data
# print("Pruned tree's accuracy on test set: {:.3f}".format(pruned_tree.score(x_test, y_test)))
# plot_tree(pruned_tree, feature_names = fn, class_names = cn, filled = True)
# plt.show()


"""PART FOUR"""
"""Regularized OLS"""
# Transform categorical variables into dummies
# Existing construction type, Proposed construction type, Supervisor District, Zipcode, Quarter Filed, Day of Week
finalOLS = pd.get_dummies(final, columns=['Existing Construction Type', 'Proposed Construction Type',
                                           'Supervisor District', 'Zipcode', 'Quarter Filed', 'Day of Week'])

# Make sample (MemoryError otherwise)
sample_OLS = finalOLS.sample(n=1000)
sample_OLS.shape

# Split data into test and train
OLS_y = sample_OLS.Process_time_weeks
OLS_x = sample_OLS.drop(['Process_time_weeks'], axis = 1).astype('float64')
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(OLS_x, OLS_y, test_size = 0.20, random_state=46)

# Baseline OLS model
linreg = LinearRegression()
model_simple = linreg.fit(xtrain2, ytrain2)
y_pred2 = model_simple.predict(xtest2)
# Performance metric (mean squared error)
ols_mse = np.sqrt(mean_squared_error(ytest2, y_pred2))
print(ols_mse)

# Elastic net (accounts for interaction between features and highly non-linear decision boundaries)
enet = ElasticNet(normalize=True)
model_regular = enet.fit(xtrain2, ytrain2)
# Estimate MSE
y_pred3 = model_regular.predict(xtest2)
enet_mse = np.sqrt(mean_squared_error(ytest2,y_pred3))
print(enet_mse)

# Tuning parameters
# Set large range of lambdas (alpha)
tuning_param = 10**np.linspace(10,-2,100)*0.5
enet_params = {"l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "alpha":[tuning_param.all()]}
enet_cv = GridSearchCV(model_regular, enet_params, cv = 10).fit(OLS_x, OLS_y)
enet_cv.best_params_

# fit the final model with the optimal alpha
# enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)
# y_pred = enet_tuned.predict(X_test)
# enet_mse_tuned = np.sqrt(mean_squared_error(y_test,y_pred))
# enet_mse_tuned
# see which coefficients were kept by ENet
# pd.Series(enet_tuned.coef_, index = X.columns)


"""PART FIVE"""
"""Random Forest"""
# Swarm intelligence and feature selection

# scale the features
sc = StandardScaler()
xtrain4 = sc.fit_transform(xtrain_s)
ytrain4 = sc.transform(xtest_s)

