#Importing necessary Libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

#Reading files
train_data=pd.read_csv(os.getcwd() +'\Customer_trainfile.csv')
test=pd.read_csv(os.getcwd() + '\Customer_testfile.csv')

print(train_data.shape) #(245725, 11)
print(test.shape) #(105312, 10)

#checking datatypes-----Age,Vintage,Avg_acc_bal,is_lead are int, remaining are object
print(train_data.info())
print(test.info())

print(train_data.nunique())
print(test.nunique())

#Descriptive stats for Numeric Variables(train_data)
NumStats=train_data.describe()
pd.set_option('display.max_columns',None)
print(NumStats)

#Descriptive stats for Numeric Variables(test)
NumStats=test.describe()
print(NumStats)

#Checking Missing values---Credit_Product has missing Values
print(train_data.isnull().sum()) #29325 missing values
print(test.isnull().sum()) #12522 missing values

print(train_data['Credit_Product'].value_counts()) # No: 144357 , Yes: 72043
print(test['Credit_Product'].value_counts()) # No: 61608 , Yes: 31182

#Filling the missing Values
train_data['Credit_Product'].replace(np.nan,'Yes',inplace=True)
print(train_data.isnull().sum())
test['Credit_Product'].replace(np.nan,'Yes',inplace=True)
print(test.isnull().sum())


#EDA---Exploratory Data Analysis(Univariate , Bivariate)
#Train_data analysis
#Target variable
sns.countplot(train_data['Is_Lead'])
plt.title('Is_Lead', fontsize = 10)
fig1=plt.show()

#Age Variable
train_data.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
fig2=plt.show()
#concl: Most of the customers of the bank in this dataset are in the age range of 20-30.

# Gender vs Is_Lead
sns.countplot(x='Gender',hue='Is_Lead',data=train_data,palette='husl')
fig3=plt.show()
# Conclusion: Customers who bought credit card are mostly Males ,Less number of females have bought Credit Card

#Occuption vs Is_Lead
sns.countplot(x='Occupation',hue='Is_Lead',data=train_data,palette='husl')
fig4=plt.show()
#Concl: Mostly self employed customers has bought credit card and enterpreneurs the least

# Credit_Product vs Occupation
sns.countplot(x='Credit_Product',hue='Occupation',data=train_data,palette='husl')
fig5=plt.show()
# Nearly 33000 selfEmployed customers have active credit product so there is a possiblity that self employed customers may buy another credit card

# Is_active vs Is_Lead
sns.countplot(x='Is_Active',hue='Is_Lead',data=train_data,palette='husl')
fig6=plt.show()
#concl: Approx 25000 or 20-21% customers are active who are interested to buy credit card(lead)
#and 30000-32000 or approx 25-27% customers are not active but interested to buy credit card(lead)

# #Vintage variable analysis
plt.figure(figsize=(13,7))
plt.subplot(2,1,1)
sns.distplot(train_data['Vintage'], color='green')
plt.title("Vintage")
fig7=plt.show()

# test data Analysis
#Age Variable
test.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
fig2a=plt.show()

# Credit_Product vs Occupation
sns.countplot(x='Credit_Product',hue='Occupation',data=test,palette='husl')
fig5a=plt.show()

# #Vintage variable analysis
plt.figure(figsize=(13,7))
plt.subplot(2,1,1)
sns.distplot(test['Vintage'], color='green')
plt.title("Vintage")
fig7a=plt.show()


# Checking Avg_Account_Balance for Skewness and removing skewness.
#Train
sns.distplot(train_data['Avg_Account_Balance'])
fig8=plt.show()
train_data['Avg_Account_Balance'] = train_data['Avg_Account_Balance'].map(lambda i: np.log(i) if i > 0 else 0)
sns.distplot(train_data['Avg_Account_Balance'])
fig8ab=plt.show()
#Test
sns.distplot(test['Avg_Account_Balance'])
fig9=plt.show()
test['Avg_Account_Balance'] = np.log(test['Avg_Account_Balance'])
sns.distplot(train_data['Avg_Account_Balance'])
fig9ab=plt.show()

#Dropping the ID column
train_data.drop(['ID'],axis=1,inplace=True)
test.drop(['ID'],axis=1, inplace=True)

print(train_data.dtypes)
print(test.dtypes)

#Converting Categorical Var to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
colname= ['Gender','Region_Code','Occupation','Channel_Code','Credit_Product','Is_Active']
for a in colname:
    train_data[a] = le.fit_transform(train_data[a])
    test[a] = le.fit_transform(test[a])
print(train_data)
print(test)

# Applying Cross validation and calculating roc-auc score
# Removing target var 'Is_Lead' from train_data and storing in another var
X=train_data.drop('Is_Lead',axis=1)
y=train_data['Is_Lead']

from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.metrics import roc_auc_score


def cross_val(X, y, model, params, folds=9):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=21)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold: {fold}")
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        algo = model(**params)
        algo.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100, verbose=400)
        pred = algo.predict_proba(x_test)[:, 1]
        roc_score = roc_auc_score(y_test, pred)
        print(f"roc_auc_score: {roc_score}")
        print("-" * 50)
    return algo

#Light Gradient Boosting Algorithm
lgbm_params = {'learning_rate': 0.1,'n_estimators': 20000, 'max_bin': 94,'num_leaves': 12,'max_depth': 30,'reg_alpha': 8.457,
              'reg_lambda': 6.853,'subsample': 1.0}
from lightgbm import LGBMClassifier
lgb_model = cross_val(X, y, LGBMClassifier, lgbm_params)
predict_test_lgb=lgb_model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'Is_Lead': predict_test_lgb})
submission.to_csv('C:/Users/hp/Desktop/submission_LgbmClassifier1.csv',index=False)
#roc_auc_score: 0.8525869220635909


#XtraGradient Boosting Algorithm
xgbm_params= {'n_estimators': 20000,  'max_depth': 5,  'learning_rate': 0.03,  'reg_lambda': 29.326, 'subsample': 0.818,
             'colsample_bytree': 0.235, 'colsample_bynode': 0.81,  'colsample_bylevel': 0.453}
from xgboost import XGBClassifier
xgbm_model = cross_val(X, y, XGBClassifier, xgbm_params)
predict_test_xgbm=xgbm_model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'Is_Lead': predict_test_xgbm})
submission.to_csv('C:/Users/hp/Desktop/submission_xgbmClassifier.csv',index=False)
#roc_auc_score: 0.8484759668139442








