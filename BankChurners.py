#Importing necessary libraries

import numpy as np # linear algebra
import pandas as pd #for data processing/handling csv file 
from matplotlib import pyplot as plt #for ploting charts
import seaborn as sns #visualization/for ploting heatmap
sns.set(style='darkgrid')
from sklearn.preprocessing import LabelEncoder #encoding categories from strings to mumbers
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV #fitting model with different parameters using k fold cross validation
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold #divide folds in uniform way
from sklearn.model_selection import train_test_split 
#from sklearn.model_selection import cross_val_score

#Main Program

#Dataset Review
df = pd.read_csv("BankChurners.csv",sep=';') #open csv file and save it to dataframe
df.shape #show dataframe dimensions(10127 rows ,11 columns)
df.isna().sum() #count null values of each column
df.nunique()#show unique values of each column
df.dtypes #show the data type of each column
df.groupby('Attrition_Flag').mean() #General correlation between coefficients and predict values

#Exploratory Data Analysis
df['Attrition_Flag'].value_counts(normalize=True)
dummies = pd.get_dummies(df.Attrition_Flag) #Transform string values to numbers 
churn = pd.concat([df,dummies],axis='columns') #keep only necessary columns
churn = churn.drop(['Attrition_Flag','Attrited Customer'], axis='columns')

#Categorical attributes
churn[['Existing Customer','Gender','Marital_Status']].groupby(['Gender','Marital_Status']).mean().round(2)
#Females slightly more likely to churn
pd.crosstab(churn['Gender'],churn['Existing Customer']).plot(kind='bar')
churn['Gender'].value_counts()

churn[['Existing Customer','Education_Level']].groupby(['Education_Level']).agg(['mean','count']).round(2)
#Customers with a doctorate degree has the highest average churn rate

churn[['Existing Customer','Card_Category']].groupby(['Card_Category']).agg(['mean','count']).round(2)
pd.crosstab(churn['Card_Category'],churn['Existing Customer']).plot(kind='bar')
#Blue category seems to be the most popular, since it is the most affordable
churn[['Existing Customer','Income_Category']].groupby(['Income_Category']).agg(['mean','count']).round(2)
pd.crosstab(churn['Income_Category'],churn['Existing Customer']).plot(kind='bar')
#Wide majority of customers have <40K $ salary, that's why they choose blue card
#People with higher salary don't need debit card, since they can pay with cash

#Numerical attributes
corr = churn.corr().round(2)#Inspect correlation between coefficients and target value(Existing Customer)
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="YlGnBu")
corr_target=abs(corr['Existing Customer']).sort_values(ascending=False)

churn[['Existing Customer','Months_Inactive_12_mon']].groupby(['Months_Inactive_12_mon']).agg(['mean','count']).round(2)
#There is a positive correlation between the number of inactive months and customer churn. 
# We clearly see that the churn rate increases as the number of inactive months increases (excluding the categories with very few customers)
value_counts = churn['Months_Inactive_12_mon'].value_counts()
to_remove = value_counts[value_counts <= 1000].index
churn = churn[~df.Months_Inactive_12_mon.isin(to_remove)]
churn[['Existing Customer','Months_Inactive_12_mon']].groupby(['Months_Inactive_12_mon']).agg(['mean','count']).round(2)
pd.crosstab(churn['Months_Inactive_12_mon'],churn['Existing Customer']).plot(kind='bar')

churn[['Existing Customer','Customer_Age','Dependent_count']].groupby(['Existing Customer']).mean().round(2)
pd.crosstab(churn['Customer_Age'],churn['Existing Customer']).plot(kind='bar')
#Customer age has same distribution around mean value(46), so customer age is not an important factor of our research

pd.crosstab(churn['Dependent_count'],churn['Existing Customer']).plot(kind='bar')
#Neither dependent count is a useful attribute

churn[['Existing Customer','Months_on_book']].groupby(['Existing Customer']).mean().round(2)
pd.crosstab(churn['Months_on_book'],churn['Existing Customer']).plot(kind='bar')

#Model Fitting
#Drop unnecessary attributes
inputs=churn.drop(['Marital_Status','CLIENTNUM','Existing Customer','Customer_Age','Dependent_count','Months_on_book'],axis='columns')
target=churn['Existing Customer']

#Encode string categories with numerical values
le_Gender = LabelEncoder()
le_Education_Level = LabelEncoder()
le_Income_Category = LabelEncoder()
le_Card_Category = LabelEncoder()
inputs['Gender'] = le_Gender.fit_transform(inputs['Gender'])
inputs['Education_Level'] = le_Education_Level.fit_transform(inputs['Education_Level'])
inputs['Income_Category'] = le_Income_Category.fit_transform(inputs['Income_Category'])
inputs['Card_Category'] = le_Card_Category.fit_transform(inputs['Card_Category'])
#Gender:Male=1,Female=0,Education_Level:High School=3,Graduate=2,Uneducated=5,Unknown=6,Post-Graduate=4,Doctorate=1,College=0,
#Income_Category:<40k =4, Card_Category:Blue=0,Gold=1,Silver=3,Platinum=2

X=inputs
Y=target

#parameter tuning in order to find the best solution
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X, Y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df) #display best permutation for each model
plt.show() #display histograms computed above


#I chose randomly logistic regression to test it with critical values for the attributes

folds = StratifiedKFold(n_splits=5)
for train_index, test_index in folds.split(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model2=LogisticRegression(solver='liblinear',multi_class='auto')
model2.fit(X_train, Y_train)
y_predicted=model2.predict(X_test)

mod1=model2.predict([[0,1,4,0,5]])#dokimi gia Gynaika me Doctorate, mistho<40K, katigoria kartas Blue, 5 mhnes anenergh
mod2=model2.predict([[0,1,4,0,1]])#dokimi gia Gynaika me Doctorate, mistho<40K, katigoria kartas Blue, 1 mhnas anenergh
mod3=model2.predict([[1,1,4,0,1]])#dokimi gia Antra me Doctorate, mistho<40K, katigoria kartas Blue, 1 mhnas anenergos
mod4=model2.predict([[1,1,4,0,5]])#dokimi gia Antra me Doctorate, mistho<40K, katigoria kartas Blue, 5 mhnes anenergos
print("Existing Customer value is :",mod1)
print("Existing Customer value is :",mod2)
print("Existing Customer value is :",mod3)
print("Existing Customer value is :",mod4)
plt.show() #display histograms computed above