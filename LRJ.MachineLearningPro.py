#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
plt.rc('font',family='SimHei',size=13)
data_trainpath=open(r'D:\GoogleDownload\train.csv')
data_train=pd.read_csv(data_trainpath)
data_train


# In[3]:


data_testpath=open(r'D:\GoogleDownload\test.csv')
data_test=pd.read_csv(data_testpath)
data_test


# In[4]:


data_train.info()#训练集


# In[5]:


data_test.info()#测试集


# In[6]:


data_train['Survived'].value_counts(normalize=True)


# In[7]:


data_train.describe()


# In[8]:


data_test.describe()


# In[9]:


a=data_train[['Sex','Survived']].groupby(['Sex']).mean()
a.columns=['存活率']
a


# In[10]:


data_train.groupby(['Sex','Survived'])['Sex'].count()


# In[11]:


a=data_train[['Embarked','Survived']].groupby(['Embarked']).mean()
a.columns=['存活率']
a


# In[12]:


data_train.groupby(['Embarked','Survived'])['Embarked'].count()


# In[13]:


a=data_train[['Pclass','Survived']].groupby(['Pclass']).mean()
a.columns=['存活率']
a


# In[14]:


data_train.groupby(['Pclass','Survived'])['Pclass'].count()


# In[15]:


pd.set_option('max_row',2000)
a=data_train[['Age','Survived']].groupby(['Age']).count()
a.columns=['总人数']
a


# In[16]:


a= data_train.Age.value_counts()
plt.xlabel('年龄')
plt.ylabel('人数')
plt.bar(a.index,a.values)


# In[17]:


a=data_train[['Age','Survived']].groupby(['Age']).mean()
a.columns=['存活率']
a


# In[18]:


a=data_train[['SibSp','Survived']].groupby(['SibSp']).mean()
a.columns=['存活率']
a


# In[19]:


a=data_train[['Parch','Survived']].groupby(['Parch']).mean()
a.columns=['存活率']
a


# In[20]:


sc=data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
sc#有客舱的乘客死亡人数和存活人数


# In[21]:


sc=data_train.Survived[pd.notnull(data_train.Cabin)].mean()

sc#有客舱的乘客存活率


# In[22]:


sc=data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
sc#无客舱的乘客死亡人数和存活人数


# In[23]:


sc=data_train.Survived[pd.isnull(data_train.Cabin)].mean()

sc#无客舱的乘客存活率


# In[24]:


a=data_train[['Fare','Survived']]
b=a.groupby(['Survived'])
df=dict([x for x in b])[1]
df
df.mean()


# In[25]:


a=data_train[['Fare','Survived']]
b=a.groupby(['Survived'])
df=dict([x for x in b])[0]
df
df.mean()


# In[26]:


data_train['Fare'].describe()


# In[27]:


a=data_train[['Fare','Survived']].groupby(['Fare']).mean()
a.columns=['存活率']
a


# In[29]:


Train_Test= data_train.append(data_test,sort=False)#将训练集和测试集合并，以方便数据清洗
Train_Test.info()


# In[30]:


Train_Test.Age = Train_Test.Age.fillna(Train_Test.Age.mean())
Train_Test.info()


# In[31]:


a=Train_Test.Embarked.value_counts()
plt.bar(a.index,a.values)


# In[32]:


Train_Test.Embarked=Train_Test.Embarked.fillna('S')
Train_Test.info()


# In[33]:


Train_Test.Fare=Train_Test.Fare.fillna(data_train.Fare.mean())
Train_Test.info()


# In[34]:


Train_Test.Cabin = Train_Test.Cabin.fillna('L')#用L填充确实Cabin值
Train_Test.Cabin=Train_Test.Cabin.apply(lambda x: x[0])
Train_Test.info()


# In[35]:


Train_Test


# In[36]:


sexdict = {'male':1, 'female':0}
Train_Test.Sex = Train_Test.Sex.map(sexdict)


# In[37]:


Train_Test.head(1)


# In[38]:


a= pd.get_dummies(Train_Test.Embarked, prefix = 'Embarked')
Train_Test = pd.concat([Train_Test,a], axis = 1) 
Train_Test.drop(['Embarked'], axis = 1, inplace=True) 
Train_Test.head(1)


# In[39]:


a = pd.get_dummies(Train_Test.Cabin, prefix = 'Cabin')
Train_Test = pd.concat([Train_Test,a], axis = 1)
Train_Test.drop(['Cabin'], axis = 1, inplace=True)
Train_Test.head(1)


# In[40]:


a = pd.get_dummies(Train_Test.Pclass, prefix = 'Pclass')
Train_Test = pd.concat([Train_Test,a], axis = 1)
Train_Test.drop(['Pclass'], axis = 1, inplace=True)
Train_Test.head(1)


# In[41]:


titleDict = {   "Capt":       "Officer", "Col":        "Officer",
               "Major":      "Officer","Jonkheer":   "Royalty",
               "Don":        "Royalty","Sir" :       "Royalty",
               "Dr":         "Officer", "Rev":        "Officer",
               "the Countess":"Royalty","Dona":       "Royalty",
               "Mme":        "Mrs", "Mlle":       "Miss",
               "Ms":         "Mrs", "Mr" :        "Mr",
               "Mrs" :       "Mrs", "Miss" :      "Miss",
               "Master" :    "Master", "Lady" :      "Royalty"
             }
Train_Test.Name = Train_Test.Name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
Train_Test.Name = Train_Test.Name.map(titleDict)
title = pd.get_dummies(Train_Test.Name, prefix = 'title')
Train_Test = pd.concat([Train_Test,title], axis = 1)
Train_Test.drop(['Name'], axis = 1, inplace=True)
Train_Test.head(1)


# In[42]:


Train_Test['Family_Member']=Train_Test.SibSp+Train_Test.Parch+1
Train_Test.head(1)


# In[43]:


Train_Test.drop(['Ticket'], axis = 1, inplace=True)
Train_Test.info()


# In[44]:


Train_X = Train_Test.iloc[0:891, :]
Train_Y = Train_X.Survived
Train_X.drop(['Survived'], axis=1, inplace =True)
Test_X  = Train_Test.iloc[891:, :]
Test_X .drop(['Survived'], axis=1, inplace =True)
Test_Y = pd.read_csv(r'D:\GoogleDownload\gender_submission.csv')
Test_Y=np.squeeze(Test_Y)
Train_X .shape,Train_Y.shape,Test_X.shape, Test_Y.shape


# In[46]:


import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
plt.figure(figsize=(14,12))
sns.heatmap(Train_Test.corr(),cmap="Blues")
plt.show()


# In[56]:


predictor = ['Sex','Age','SibSp', 'Parch','Fare','Embarked_C','Cabin_B','Cabin_D','Cabin_E','title_Master', 
              'title_Miss', 'title_Mr', 'title_Mrs', 'Pclass_1']
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')


# In[57]:


model.fit(Train_X[predictor].iloc[0:-100,:],Train_Y.iloc[0:-100])


# In[58]:


from sklearn.metrics import accuracy_score
accuracy_score(model.predict(Train_X[predictor].iloc[-100:,:]),Train_Y.iloc[-100:].values.reshape(-1, 1))


# In[420]:


prediction = model.predict(Test_X[predictor])
conse = pd.DataFrame({'PassengerId':Test_Y['PassengerId'].values, 'Survived':prediction.astype(np.int32)})
conse.to_csv('D:\GoogleDownload\LRJAfterHandling.csv', index=False)
conse.head(10)






