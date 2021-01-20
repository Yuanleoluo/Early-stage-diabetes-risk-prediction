#!/usr/bin/env python
# coding: utf-8

# # Early stage diabetes risk prediction

# 1. Introduction
# 2. Data
# 3. EDA
# 4. Data Engineering/cleaning
# 5. Model Building
# 6. Test

# <h1>1. Introduction

# Would it be nice if we can find the early signs of diabetes? One of the promising application of machine learning can deliver just that. While this concept is nothing new and is probably already widely used in many medical fields, it could be very helpful for begging data scientists to see the work flow with different datasets and style. Let's get to it.

# <h1>2. Data

# I acquire the data from University of California Irvine(https://archive.ics.uci.edu/ml/index.php) already in CSV format. Its size is relatively very small, 520 instances(rows) with 17 attributes(columns). The data was donated to UCI fairly recently on July 12th, 2020. Although metadata wrote that the data consists of missing values, I did not find any as you will see later.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[27]:


df = pd.read_csv('../diabetes_data.csv')


# <h1> 3. EDA

# In[3]:


df.head()


# In[6]:


df.shape


# In[6]:


# missing values and datatype
df.info()


# In[8]:


df.isnull().sum()


# In[4]:


# graph of target variable
sns.countplot(x="class", data=df, palette = "Set2")

# table of target variable
df['class'].value_counts(normalize=True).to_frame()


# In[38]:


sns.distplot(df['Age'])
print('Skewness: '+ str(df['Age'].skew())) 
print("Kurtosis: " + str(df['Age'].kurt()))


# kurtosis and skewness are in acceptable range.

# In[17]:


sns.boxplot(x="class", y="Age", data=df)


# In[18]:


df.groupby(['Gender'])['class'].value_counts(normalize=True).to_frame()


# In[19]:


pd.crosstab(df.Gender, df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[25]:


df.groupby(['Obesity'])['class'].value_counts(normalize=True).to_frame()


# In[26]:


pd.crosstab(df['Obesity'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[24]:


df.groupby(['muscle stiffness'])['class'].value_counts(normalize=True).to_frame()


# In[50]:


pd.crosstab(df['muscle stiffness'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[27]:


df.groupby(['Alopecia'])['class'].value_counts(normalize=True).to_frame()


# In[28]:


pd.crosstab(df['Alopecia'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[29]:


df.groupby(['partial paresis'])['class'].value_counts(normalize=True).to_frame()


# In[30]:


pd.crosstab(df['partial paresis'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[31]:


df.groupby(['delayed healing'])['class'].value_counts(normalize=True).to_frame()


# In[32]:


pd.crosstab(df['delayed healing'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[33]:


df.groupby(['Irritability'])['class'].value_counts(normalize=True).to_frame()


# In[34]:


pd.crosstab(df['Irritability'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[35]:


df.groupby(['Itching'])['class'].value_counts(normalize=True).to_frame()


# In[37]:


pd.crosstab(df['Itching'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[36]:


df.groupby(['visual blurring'])['class'].value_counts(normalize=True).to_frame()


# In[38]:


pd.crosstab(df['visual blurring'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[39]:


df.groupby(['Genital thrush'])['class'].value_counts(normalize=True).to_frame()


# In[40]:


pd.crosstab(df['Genital thrush'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[41]:


df.groupby(['Polyphagia'])['class'].value_counts(normalize=True).to_frame()


# In[42]:


pd.crosstab(df['Polyphagia'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[43]:


df.groupby(['weakness'])['class'].value_counts(normalize=True).to_frame()


# In[44]:


pd.crosstab(df['weakness'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[45]:


df.groupby(['sudden weight loss'])['class'].value_counts(normalize=True).to_frame()


# In[46]:


pd.crosstab(df['sudden weight loss'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[47]:


df.groupby(['Polydipsia'])['class'].value_counts(normalize=True).to_frame()


# In[48]:


pd.crosstab(df['Polydipsia'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# In[49]:


df.groupby(['Polyuria'])['class'].value_counts(normalize=True).to_frame()


# In[50]:


pd.crosstab(df['Polyuria'], df['class'], margins=True).style.background_gradient(cmap='summer_r')


# <h1>Data Engineering/Cleaning

# In[28]:


numeric_feats = df.dtypes[df.dtypes != "object"].index

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_feats] = scaler.fit_transform(df[numeric_feats])      #standardscaler transform the Age column


# In[29]:


df['class'] = df['class'].replace({'Negative': 0, 'Positive': 1})  #replace string to binary 0/1
df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
df['Polyuria'] = df['Polyuria'].replace({'Yes': 1, 'No': 0})
df['Polydipsia'] = df['Polydipsia'].replace({'Yes': 1, 'No': 0})
df['sudden weight loss'] = df['sudden weight loss'].replace({'Yes': 1, 'No': 0})
df['weakness'] = df['weakness'].replace({'Yes': 1, 'No': 0})
df['Polyphagia'] = df['Polyphagia'].replace({'Yes': 1, 'No': 0})
df['Genital thrush'] = df['Genital thrush'].replace({'Yes': 1, 'No': 0})
df['visual blurring'] = df['visual blurring'].replace({'Yes': 1, 'No': 0})
df['Irritability'] = df['Irritability'].replace({'Yes': 1, 'No': 0})
df['partial paresis'] = df['partial paresis'].replace({'Yes': 1, 'No': 0})
df['muscle stiffness'] = df['muscle stiffness'].replace({'Yes': 1, 'No': 0})
df['Alopecia'] = df['Alopecia'].replace({'Yes': 1, 'No': 0})
df['Obesity'] = df['Obesity'].replace({'Yes': 1, 'No': 0})


# In[30]:


df_train = df.sample(frac = 0.8)    #train/cross-val
df_test = df.drop(df_train.index)   #test


# In[31]:


X = df_test.drop(columns=['Itching', 'delayed healing', 'class'])     #independent variable for testing
x = df_train.drop(columns=['Itching', 'delayed healing', 'class'])    #independent variable for training/cross-validation
x


# In[32]:


Y = df_test['class'].values     #target variable for testing
y = df_train['class'].values    #target variable for training/cross-validation
y


# # Model Building

# In[33]:


# Root mean square error by juliencs from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

scorer = make_scorer(mean_squared_error, greater_is_better = False)
def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)


# <h3>Decision Tree

# In[34]:


from sklearn import tree


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# In[36]:


tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X_train, y_train)


# In[37]:


tree_model.score(X_test, y_test)


# In[38]:


scores = cross_val_score(tree_model, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("RMSE on Training set :", rmse_cv_train(tree_model).mean())


# <h3> Logistic Reg

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)


# In[41]:


LR_model.score(X_test, y_test)


# In[ ]:


scores = cross_val_score(LR_model, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("RMSE on Training set :", rmse_cv_train(LR_model).mean())


# <h3>KNN

# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# In[18]:


KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)


# In[19]:


KNN_model.score(X_test, y_test)


# In[20]:


scores = cross_val_score(KNN_model, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("RMSE on Training set :", rmse_cv_train(KNN_model).mean())


# <h3>SVM

# In[21]:


from sklearn import svm


# In[22]:


svm_model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
svm_model.fit(X_train, y_train)


# In[23]:


svm_model.score(X_test, y_test)


# In[24]:


scores = cross_val_score(svm_model, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("RMSE on Training set :", rmse_cv_train(svm_model).mean())


# <h1>Test

# In[25]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# knn F1&jaccard score
knn_yhat = KNN_model.predict(X)
knn_f1 = round(f1_score(Y, knn_yhat, average='weighted'), 2)
knn_jaccard = round(jaccard_score(Y, knn_yhat), 2)

# Decision Tree F1&jaccard score
DT_yhat = tree_model.predict(X)
DT_f1 = round(f1_score(Y, DT_yhat, average='weighted'), 2)
DT_jaccard = round(jaccard_score(Y, DT_yhat), 2)

# Support Vector Machine Tree F1&jaccard score
svm_yhat = svm_model.predict(X)
svm_f1 = round(f1_score(Y, svm_yhat, average='weighted'), 2)
svm_jaccard = round(jaccard_score(Y, svm_yhat), 2)

# Logistic Regression F1&jaccard& logloss score
LR_yhat = LR_model.predict(X)
LR_prob = LR_model.predict_proba(X)
LR_f1 = round(f1_score(Y, LR_yhat, average='weighted'), 2)
LR_jaccard = round(jaccard_score(Y, LR_yhat), 2)

# log loss
loss = round(log_loss(Y, LR_prob), 2)


# In[26]:


# display reports
df_report = pd.DataFrame(np.array([['KNN', knn_jaccard, knn_f1,'NA'], 
                                   ['Decision Tree',DT_jaccard, DT_f1,'NA'], 
                                   ['SVM',svm_f1, svm_f1,'NA'],
                                   ["LogisticRegression", LR_jaccard, LR_f1, loss]]),
                           columns=['Algorithm', 'Jaccard', 'F1-score', "LogLoss"])
df_report.set_index('Algorithm')


# In[ ]:




