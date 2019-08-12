#importing libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from  matplotlib import pyplot



#reading the csv file through pands
dataset = pd.read_csv('day.csv')
bike_rent=dataset

##exploratory data analysis
dataset['season']= dataset['season'].astype('category')
dataset['yr']=dataset['yr'].astype('int')
dataset['mnth']=dataset['mnth'].astype('category')
dataset['holiday']=dataset['holiday'].astype('int')
dataset['workingday']=dataset['workingday'].astype('int')
dataset['weekday']=dataset['weekday'].astype('category')
dataset['weathersit']=dataset['weathersit'].astype('category')
d1=dataset['dteday'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], '%Y-%m-%d').strftime('%d')
dataset['dteday']=d1
dataset['dteday']=dataset['dteday'].astype('category')
dataset = dataset.drop(['instant','casual', 'registered'], axis=1)

#Creating dataframe with missing values present in each variable
missing_val = pd.DataFrame(dataset.isnull().sum()).reset_index()

##Outlair Analysis

#saving numeric values#
clnames=["temp","atemp","hum","windspeed",]
#ploting boxplot visualize outliers
plt.boxplot(dataset['temp'])

plt.boxplot(dataset('atemp'))
plt.boxplot(dataset("hum"))
plt.boxplot(dataset("windspeed"))

##Feature Selection
#setting width and height of the plot
aa ,bb =plt.subplots(figsize=(7,5))
#setting width and height of the plot
aa ,bb =plt.subplots(figsize=(7,5))
#generate correlation matrix
corr = pd.DataFrame.corr(dataset)

#Plot using seaborn library
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#removing correlated variable
dataset=dataset.drop(['atemp'], axis=1)


##Model devlopment
#splitting the data into train and test
train, test = train_test_split(dataset, test_size=0.2)

##DecissionTree
DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:11], train.iloc[:,11])
prediction_DT = DT.predict(test.iloc[:,0:11])

##RandomForest
RF = RandomForestRegressor(n_estimators = 500).fit(train.iloc[:,0:11], train.iloc[:,11])
RF_predictions = RF.predict(test.iloc[:,0:11])


#defining MAPE function
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

#MAPE for decision tree regression
MAPE(test.iloc[:,11], prediction_DT)
#mape=36.00

#mape for randomforest
MAPE(test.iloc[:,11],RF_predictions)
#mape=18.19


#writting the result to a csvfile

result=pd.DataFrame(test.iloc[:,0:11])
result['pred_cnt'] = (RF_predictions)

result.to_csv("Randomforest_python.csv",index=False)
