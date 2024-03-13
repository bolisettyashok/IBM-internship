#!/usr/bin/env python
# coding: utf-8

# ## The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD). The dataset provides the patients’ information. It includes over 3390 records and 15 attributes.
# 

# # **Variables**
# Each attribute is a potential risk factor. There are both demographic behavioral, and medical risk factors.
# 

# # **Data Description**
# Demographic:
# 
# 
# *   Sex: male or female("M" or "F")
# *   Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous) Behavioral
# *   Is_smoking: whether or not the patient is a current smoker ("YES" or "NO")
# *   Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.) Medical( history)
# *   BP Meds: whether or not the patient was on blood pressure medication (Nominal)
# *   Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
# *   Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
# *   Diabetes: whether or not the patient had diabetes (Nominal)
# Medical(current)
# *   Tot Chol: total cholesterol level (Continuous)
# *   Sys BP: systolic blood pressure (Continuous)
# *   Dia BP: diastolic blood pressure (Continuous)
# *   BMI: Body Mass Index (Continuous)
# *   Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in
# fact discrete, yet are considered continuous because of large number of possible values.)
# *   Glucose: glucose level (Continuous)
# Predict variable (desired target)
# *   **10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”) - DV**
# 
# 
# 
# 

# #**Loading and Exploring Data**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe(include='all').T


# Checking For duplication of Data.

# In[8]:


df[df.duplicated()]


# #<b> Handling Missing Values

# 
# 
# Checking for Missing Values

# In[9]:


df.isnull().sum()


# In[10]:


# Before Altering the Data copying orinal data
df_copy=df.copy()


# **Hear total Missing data was less than 13% insted of delecting data we replacing missing values with approxmate values**

# In[11]:


# Missing Value Count Function
def show_missing():
    missing = df_copy.columns[df_copy.isnull().any()].tolist()
    return missing

# Missing data counts and percentage
print('Missing Data Count')
print(df_copy[show_missing()].isnull().sum().sort_values(ascending = False))
print('--'*50)
print('Missing Data Percentage')
print(round(df_copy[show_missing()].isnull().sum().sort_values(ascending = False)/len(df_copy)*100,2))


# In[12]:


round(df_copy[show_missing()].isnull().sum().sort_values(ascending = False)/len(df_copy)*100,2).plot(kind='bar', color=['red', 'yellow', 'blue', 'orange'])


# **Glucose**
# 
# In the following column mean and median are nearby. we have fill the missing values so i am using median values for filling the missing values.

# In[13]:


df['glucose'].describe()


# In[14]:


print('Glucose Feature Missing Before')
print(df_copy[['glucose']].isnull().sum())
print('--'*50)
df_copy['glucose']=df_copy['glucose'].fillna(df['glucose'].median())
print('Glucose Feature Missing After')
print(df_copy[['glucose']].isnull().sum())
print('--'*50)


# **Education**

# In[15]:


df['education'].describe()


# In[16]:


df['education'].unique()


# Education feature is not a continues variable so we using Mode for filling the missing values.

# In[17]:


print('Education Feature Missing Before')
print(df_copy[['education']].isnull().sum())
print('--'*50)
df_copy['education']=df_copy['education'].fillna(df['education'].mode()[0])
print('Education Feature Missing After')
print(df_copy[['education']].isnull().sum())
print('--'*50)


# **BPMeds**

# In[18]:


df['BPMeds'].describe()


# In[19]:


df['BPMeds'].unique()


# In[20]:


print('BPMeds Feature Missing Before')
print(df_copy[['BPMeds']].isnull().sum())
print('--'*50)
df_copy['BPMeds']=df_copy['BPMeds'].fillna(df['BPMeds'].mode()[0])
print('BPMeds Feature Missing After')
print(df_copy[['BPMeds']].isnull().sum())
print('--'*50)


# Total Cholostral

# In[21]:


df['totChol'].describe()


# In[22]:


print('Total colostrol Feature Missing Before')
print(df_copy[['totChol']].isnull().sum())
print('--'*50)
df_copy['totChol']=df_copy['totChol'].fillna(df['totChol'].median())
print('Total colostrol Feature Missing After')
print(df_copy[['totChol']].isnull().sum())
print('--'*50)


# **Cigrates per Day**

# In[23]:


df['cigsPerDay'].describe()


# In[24]:


print('Cigars per day Feature Missing Before')
print(df_copy[['cigsPerDay']].isnull().sum())
print('--'*50)
df_copy['cigsPerDay']=df_copy['cigsPerDay'].fillna(df['cigsPerDay'].median())
print('Cigars per day Feature Missing After')
print(df_copy[['cigsPerDay']].isnull().sum())
print('--'*50)


# **Body Mass Index(BMI)**

# In[25]:


df['BMI'].describe()


# In[26]:


print('BMI Feature Missing Before')
print(df_copy[['BMI']].isnull().sum())
print('--'*50)
df_copy['BMI']=df_copy['BMI'].fillna(df['BMI'].median())
print('BMI Feature Missing After')
print(df_copy[['BMI']].isnull().sum())
print('--'*50)


# Heart Rate

# In[27]:


df['heartRate'].describe()


# In[28]:


print('Heart Rate Feature Missing Before')
print(df_copy[['heartRate']].isnull().sum())
print('--'*50)
df_copy['heartRate']=df_copy['heartRate'].fillna(df['heartRate'].median())
print('Heart Rate Feature Missing After')
print(df_copy[['heartRate']].isnull().sum())
print('--'*50)


# #<b> EDA 

# ###**Age**

# Data contains people of age from 32-70 years. People are effected to cardivasucular Desises from 35 

# In[29]:


fig, ax = plt.subplots(figsize=(15,6))
age_dis=pd.DataFrame(df.groupby(['age'])['id'].count())
sns.barplot(x=age_dis.index,y=age_dis['id'])
plt.ylabel('Counts')
plt.title('Age Distrubution')


# In[30]:


plt.rcParams['figure.figsize'] = (15, 5)
df.groupby(['age','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('Age wise Effected People')


# In[31]:


plt.rcParams['figure.figsize'] = (20, 5)
df.groupby(['age','sex'])['TenYearCHD'].count().unstack().plot(kind='bar')


# ###**Education**

# In[32]:


df['education'].unique()


# In[33]:


df.groupby(['education'])['id'].count()


# In[34]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['education'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('Education')
plt.ylabel('Counts')
plt.title('season Distribution in year')
ax2=plt.subplot(1,2,2)
df.groupby(['education'])['id'].count().plot(kind='pie')
plt.title('Education Proposanate')


# In[35]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['education','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('Education with Ten years CHD')


# ###**SEX**

# In[36]:


df.groupby(['sex'])['id'].count()


# In[37]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['sex'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('sex')
plt.ylabel('Counts')
plt.title('sex Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['sex'])['id'].count().plot(kind='pie')
plt.title('sex ratio Proposanate')


# In[38]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['sex','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('Sex distribution with Ten years CHD')


# ###**Smoking Data**

# In[39]:


df.groupby(['is_smoking'])['id'].count()


# In[40]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['is_smoking'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('is_smoking')
plt.ylabel('Counts')
plt.title('Smoking people Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['is_smoking'])['id'].count().plot(kind='pie')
plt.title('Smoking persons Proposanate')


# In[41]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['is_smoking','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('Smoking people distribution with Ten years CHD')


# In[42]:


plt.rcParams['figure.figsize'] = (15, 5)
df.groupby(['age','is_smoking'])['id'].count().unstack().plot(kind='bar')
plt.title('Age wise smoking people')


# ###**Cigretes per Day**

# In[43]:


df.groupby(['cigsPerDay'])['id'].count()


# In[44]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['cigsPerDay'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('cigsPerDay')
plt.ylabel('Counts')
plt.title('Cigretes per day Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['cigsPerDay'])['id'].count().plot(kind='pie')
plt.title('cigrets per day Proposanate')


# In[45]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['cigsPerDay','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('cig per day with Ten years CHD')


# ###**BP Medication**

# In[46]:


df.groupby(['BPMeds'])['id'].count()


# In[47]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['BPMeds'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('BPMeds')
plt.ylabel('Counts')
plt.title('BP medication people Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['BPMeds'])['id'].count().plot(kind='pie')
plt.title('BP Medication persons Proposanate')


# In[48]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['BPMeds','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('BP Medication people distribution with Ten years CHD')


# ###**Prevalent Stroke**

# In[49]:


df.groupby(['prevalentStroke'])['id'].count()


# In[50]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['prevalentStroke'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('prevalentStroke')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['prevalentStroke'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[51]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['prevalentStroke','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**Hypertension**

# In[52]:


df.groupby(['prevalentHyp'])['id'].count()


# In[53]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['prevalentHyp'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('prevprevalentHypalentStroke')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['prevalentHyp'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[54]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['prevalentHyp','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**diabetes**

# In[55]:


df.groupby(['diabetes'])['id'].count()


# In[56]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['diabetes'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('diabetes')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['diabetes'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[57]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['diabetes','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**totChol**

# In[58]:


df.groupby(['totChol'])['id'].count()


# In[59]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['totChol'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('totChol')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['totChol'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[60]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['totChol','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**sysBP**

# In[61]:


df.groupby(['sysBP'])['id'].count()


# In[62]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['sysBP'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('sysBP')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['sysBP'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[63]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['sysBP','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**diaBP**

# In[64]:


df.groupby(['diaBP'])['id'].count()


# In[65]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['diaBP'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('diaBP')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['diaBP'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[66]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['diaBP','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**BMI**

# In[67]:


df.groupby(['BMI'])['id'].count()


# In[68]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['BMI'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('BMI')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['BMI'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[69]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['BMI','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**Heart Rate**

# In[70]:


df.groupby(['heartRate'])['id'].count()


# In[71]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['heartRate'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('heartRate')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['heartRate'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[72]:


plt.rcParams['figure.figsize'] = (10, 5)
df_copy.groupby(['heartRate','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# ###**Glucose**

# In[73]:


df.groupby(['glucose'])['id'].count()


# In[74]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
ax1=plt.subplot(1,2,1)
Education_status=pd.DataFrame(df.groupby(['glucose'])['id'].count())
sns.barplot(x=Education_status.index,y=Education_status['id'])
plt.xlabel('prevalentStroke')
plt.ylabel('Counts')
plt.title('prevalent Stroke data Distribution')
ax2=plt.subplot(1,2,2)
df.groupby(['glucose'])['id'].count().plot(kind='pie')
plt.title('prevalent Stroke persons Proposanate')


# In[75]:


plt.rcParams['figure.figsize'] = (25, 5)
df_copy.groupby(['glucose','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('prevalentStroke people distribution with Ten years CHD')


# In[76]:


le=LabelEncoder()
df_copy['sex']=le.fit_transform(df_copy['sex'])
df_copy['is_smoking']=le.fit_transform(df_copy['is_smoking'])


# In[77]:


df_copy.head()


# In[78]:


df_copy.drop(['id'],axis=1,inplace=True) #Id is not useful for Model training.


# In[79]:


fig = plt.figure(figsize = (20,15))
ax = fig.gca()
df_copy.hist(ax = ax)


# In[80]:


plt.figure(figsize=(15,8))
correlation = df_copy.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# In[ ]:


for i in df_copy.columns[:-1]:
  fig = plt.figure(figsize=(9, 6))
  ax = fig.gca()
  df_copy.boxplot(column = i, by = 'TenYearCHD', ax = ax)
  ax.set_ylabel(i)
plt.show()


# #**Classfication- Mechine Learning**

# ###Data Splitting

# In[ ]:


X=df_copy[['age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose']].copy()
y=df_copy['TenYearCHD'].copy()


# In[ ]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.2, random_state = 0) 
print(X_train.shape)
print(X_test.shape)


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# ## **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(fit_intercept=True, max_iter=10000)
clf.fit(X_train, y_train)


# In[ ]:


# Get the model coefficients
clf.coef_


# In[ ]:


clf.intercept_


# In[ ]:


# Get the predicted classes
train_class_preds = clf.predict(X_train)
test_class_preds = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


# In[ ]:


# Get the accuracy scores
train_accuracy = accuracy_score(train_class_preds,y_train)
test_accuracy = accuracy_score(test_class_preds,y_test)

print("The accuracy on train data is ", train_accuracy)
print("The accuracy on test data is ", test_accuracy)


# In[ ]:


# Get the confusion matrix for both train and test
plt.figure(figsize=(5,3))
labels = ['Negative', 'Positive']
cm = confusion_matrix(y_train, train_class_preds)
print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax) #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[ ]:


# Get the confusion matrix for both train and test
plt.figure(figsize=(5,3))
labels = ['Retained', 'Churned']
cm = confusion_matrix(y_test, test_class_preds)
print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[ ]:


y_lr_predict_pro=clf.predict_proba(X_test)[:,1]


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_lr_predict_pro)


# In[ ]:


roc_auc_score(y_test,y_lr_predict_pro)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC curve')
plt.show()


# ##<b> Handling Data Imbalalance

# In[ ]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
# X_sm, y_sm = smote.fit(X,y)


# In[ ]:


y_sm=pd.DataFrame(y_sm)


# In[ ]:


y_sm.value_counts()


# ##<b> Logistic Regression

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X_sm,y_sm , test_size = 0.2, random_state = 0) 
print(X_train.shape)
print(X_test.shape)


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:


clf = LogisticRegression(fit_intercept=True, max_iter=10000)
clf.fit(X_train, y_train)


# In[ ]:


# Get the predicted classes
train_class_preds = clf.predict(X_train)
test_class_preds = clf.predict(X_test)


# In[ ]:


# Get the accuracy scores
train_accuracy = accuracy_score(train_class_preds,y_train)
test_accuracy = accuracy_score(test_class_preds,y_test)

print("The accuracy on train data is ", train_accuracy)
print("The accuracy on test data is ", test_accuracy)


# In[ ]:


# Get the confusion matrix for both train and test

cm = confusion_matrix(y_train, train_class_preds)
print('Confusion Matrix for training Data')
print(cm)
cm = confusion_matrix(y_test, test_class_preds)
print('Confusion Matrix for Test Data')
print(cm)


# In[ ]:


y_lr_predict_pro=clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_lr_predict_pro)


# In[ ]:


roc_auc_score(y_test,y_lr_predict_pro)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC curve')
plt.show()


# ##<b>Decision Tree Classifier

# In[ ]:


dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(round(dt_classifier.score(X_test, y_test),2))


# In[ ]:


plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# In[ ]:


y_dt_predict_pro=dt_classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_dt_predict_pro)


# In[ ]:


roc_auc_score(y_test,y_dt_predict_pro)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision tree ROC curve')
plt.show()


# In[ ]:


features = X.columns
importances = dt_classifier.feature_importances_
indices = np.argsort(importances)


# In[ ]:


plt.figure(figsize=(12,9))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ##<b> Random Forest

# In[ ]:


classifier = RandomForestClassifier() # For GBM, use GradientBoostingClassifier()
grid_values = {'n_estimators':[50, 65, 80, 95,120], 'max_depth':[3, 5, 7,9,12]}
GSclassifier = GridSearchCV(classifier, param_grid = grid_values, scoring = 'roc_auc', cv=5)

# Fit the object to train dataset
GSclassifier.fit(X_train, y_train)


# In[ ]:


bestvalues=GSclassifier.best_params_
GSclassifier.best_params_


# In[ ]:


classifier = RandomForestClassifier(max_depth=bestvalues['max_depth'],n_estimators=bestvalues['n_estimators']) # For GBM, use GradientBoostingClassifier()

classifier.fit(X_train, y_train)


# In[ ]:


y_train_preds_rf =  classifier.predict(X_train)
y_test_preds_rf= classifier.predict(X_test)


# In[ ]:


# Obtain accuracy on train set
accuracy_score(y_train,y_train_preds_rf)


# In[ ]:


# Obtain accuracy on test set
accuracy_score(y_test,y_test_preds_rf)


# In[ ]:


y_rf_predict_pro=classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_rf_predict_pro)


# In[ ]:


roc_auc_score(y_test,y_rf_predict_pro)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC curve')
plt.show()


# In[ ]:


features = X.columns
importances = classifier.feature_importances_
indices = np.argsort(importances)


# In[ ]:


plt.figure(figsize=(12,9))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ##<b>KNN

# In[ ]:


param_grid = {'n_neighbors':np.arange(1,50)}


# In[ ]:


knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_sm,y_sm)


# In[ ]:


bestPermet=knn_cv.best_params_
knn_cv.best_params_


# In[ ]:


#Setup arrays to store training and test accuracies
neighbors = np.arange(1,30)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model
    knn.fit(X_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 


# In[ ]:


plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


# Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=bestPermet['n_neighbors'])


# In[ ]:


# Fit the model
knn.fit(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)


# In[ ]:


y_test_pred_knn = knn.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_test_pred_knn)


# In[ ]:


y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[ ]:


roc_auc_score(y_test,y_pred_proba)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Knn(n_neighbors=2) ROC curve')
plt.show()


# ##<b> SVM

# In[ ]:


svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(round(svc_classifier.score(X_test, y_test),2))


# In[ ]:


# colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.figure(figsize=(7,4))
plt.bar(kernels, svc_scores,color=['red', 'yellow', 'blue', 'orange'])
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')


# In[ ]:


svm=SVC(probability=True)
svm.fit(X_train,y_train)


# In[ ]:


svm.score(X_test,y_test)


# In[ ]:


y_svm_predi=svm.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_svm_predi)


# In[ ]:


y_svm_predict_pro=svm.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_svm_predict_pro)


# In[ ]:


roc_auc_score(y_test,y_svm_predict_pro)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC curve')
plt.show()


# ##<b> Cat Boost

# In[ ]:


get_ipython().system('pip install catboost')


# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:


catboost=CatBoostClassifier(iterations=100,learning_rate=0.03)


# In[ ]:


catboost.fit(X_train,y_train,verbose=10)


# In[ ]:


y_catboost_pred=catboost.predict(X_test)
y_catboost_pre_prob=catboost.predict_proba(X_test)[:,1]


# In[ ]:


catboost.score(X_test,y_test)


# In[ ]:


confusion_matrix(y_test,y_catboost_pred)


# In[ ]:


roc_auc_score(y_test,y_catboost_pre_prob)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_catboost_pre_prob)


# In[ ]:


plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Cat Boost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cat Boost ROC curve')
plt.show()


# ##<b> Neural Networks
# 

# In[ ]:


X_train=pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)


# In[ ]:


# #since we shuffled, the index numbers were messed up, this resets them
X_train = X_train.reset_index(drop=True) 
y_train = y_train.reset_index(drop=True)

#convert to numpy arrays with float values
X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=float)

#reshape y_train to make matrix multiplication possible
y_train = np.array(y_train).reshape(-1, 1)


# In[ ]:


class Perceptron:
    def __init__(self, x, y):

        self.input = np.array(x, dtype=float) 
        self.label = np.array(y, dtype=float)
        self.weights = np.random.rand(x.shape[1], y.shape[1]) #randomly initialize the weights
        self.z = self.input@self.weights #dot product of the vectors
        self.yhat = self.sigmoid(self.z) #apply activation function

    
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_deriv(self, x):
        s = sigmoid(x)
        return s(1-s)

    def forward_prop(self):
        self.yhat = self.sigmoid(self.input @ self.weights) #@ symbol represents matrix multiplication (also works for vectors)
        return self.yhat

    def back_prop(self):
        gradient = self.input.T @ (-2.0*(self.label - self.yhat)*self.sigmoid(self.yhat))  #self.input is the x value

        self.weights -= gradient #process of finding the minimum loss


# In[ ]:


simple_nn = Perceptron(X_train, y_train)
training_iterations = 1000

history = [] #we will store how the mean squared error changes after each iteration in this array

def mse(yhat, y):
    sum = 0.0
    for pred, label in zip(yhat, y):
        sum += (pred-label)**2
    return sum/len(yhat)

for i in range(training_iterations):
    simple_nn.forward_prop()
    simple_nn.back_prop()
    yhat = simple_nn.forward_prop()
    history.append(mse(yhat, simple_nn.label))

    
    
yhat = simple_nn.forward_prop()
print(f'Final Mean Squared Error: {mse(yhat, simple_nn.label)}')


# In[ ]:


plt.plot(history)
plt.ylabel('Mean Squared Error')
plt.xlabel('Training Iteration')


# In[ ]:




