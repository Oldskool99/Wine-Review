# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:58:53 2020

@author: Oldskool
"""

#Wine Reviews – Code

#EDA - Code

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['figure.figsize'] = (16,8)

data = pd.read_csv('train.csv')
data.info()



data.drop(['region_1','region_2', 'designation','user_name'],axis=1,inplace=True)  
#these columns have data which is very less
data.head()


sns.countplot(x="points", data=data)
#Moist points [Average] are from 87-91

sns.distplot(data['price'], bins=500).set(xlim=(0, 250))
#Mostly prices are below 50 which is actually ideal

g = sns.countplot(x="province", data=data, order = data['province'].value_counts().iloc[:20].index )
plt.setp(g.get_xticklabels(), rotation=45)
plt.show()
#Most wine comes from California

g = sns.countplot(x="variety", data=data,  order = data['variety'].value_counts().iloc[:20].index)
plt.setp(g.get_xticklabels(), rotation=45)
plt.show()

#Pinot Noir, Chardonnay, Red Blend (red blend is a wine that’s not madefrom a specific grape variety) and Cabernet Sauvignon seems to be the  most used grape variety. 


dfWinary.sort_values(by='price', ascending=False)['price'].iloc[:20].plot(kind='bar').set_title('Winery average price')
plt.show()

dfWinary['P/P'] = dfWinary.points / dfWinary.price
dfWinary.sort_values(by='P/P', ascending=False)['P/P'].iloc[:20].plot(kind='bar').set_title('Most bang for the buck')
plt.show()
#So these wineries are the ones with best rating for the cheapest cost! 

#-----------------------------------------------------------------------------------------------------------------------------------------------------


#Random Forest Classification Code

#Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
import re 
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Reading Data
dataset = pd.read_csv("train.csv", quoting = 2)
dataset.drop(['region_1','region_2', 'designation','user_name'],axis=1,inplace=True)  
#these columns have data whisch is very less


#Cleaning Text
def corpus(s):
  review = re.sub ("[^a-zA-Z]", " ",s)
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
  review = " ".join(review)
  return review
dataset['review_description'].apply(corpus)


#Pickling Corpus
pickle_out = open("corp.pickle", "wb")
pickle.dump(corpus, pickle_out)
pickle_out.close()

#Reloading Corpus
pickle_in = open("corp.pickle", "rb")
corpus = pickle.load(pickle_in)


#Creating Bag of Word Model
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(dataset['review_description'].values).toarray()
le = LabelEncoder()
y = dataset['variety']
y = le.fit_transform(y)
sc = StandardScaler()
X=sc.fit_transform(X)

#Splitting for Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print("No of examples in train "+str(X_train.shape[0]))
print("No of examples in test "+str(X_test.shape[0]))
print("Shapes of train and test\n"+str(X_train.shape)+"\n"+str(y_train.shape)+"\n"+str(X_test.shape)+"\n"+str(y_test.shape)+"\n")

#Random Forest Classifier
RF_classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train, y_train)

#Pickling Random Forest Classifier
pickle_out = open("RF_classifier.pickle", "wb")
pickle.dump (RF_classifier, pickle_out)
pickle_out.close()

#importing Random Forest Classifier
pickle_in = open("RF_classifier.pickle", "rb")
RF_classifier = pickle.load(pickle_in)


#Evaluation of Random Forest
y_pred = RF_classifier.predict(X_test)
y_pred = le.inverse_transform(y_pred)
y_test=le.inverse_transform(y_test)
clf_r=classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

labels=le.inverse_transform(range(28))
print("Test Accuracy "+str(ac.round(4)*100)+" %")

#Random Forest Confusion Matrix 
fig, ax = plt.subplots(figsize=(28,28))
sns.set(font_scale=1.8)  
sns.heatmap(cm,annot=True,square=True,cbar=False,xticklabels=labels,yticklabels=labels, linewidths=.5,fmt='g')

plt.title('Confusion Matrix')	


#Random Forest Classification Report 
print("Classification report")
print(clf_r)

#Model Accuracy = 64%

#-----------------------------------------------------------------------------------------------------------------------------------

#Support Vector Machine (SVM) Classification

#SVM Classifier 
SVM_classifier = SVC(kernel = 'linear', random_state = 0)
SVM_classifier.fit(X_train, y_train)

#Pickling SVM Classifier
pickle_out = open("SVM_classifier.pickle", "wb")
pickle.dump (SVM_classifier, pickle_out)
pickle_out.close()

#importing SVM Classifier
pickle_in = open("SVM_classifier.pickle", "rb")
SVM_classifier = pickle.load(pickle_in)

#Evaluation of SVM
y_pred = RF_classifier.predict(X_test)
y_pred = le.inverse_transform(y_pred)
y_test=le.inverse_transform(y_test)
clf_r=classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

labels=le.inverse_transform(range(28))
print("Test Accuracy "+str(ac.round(4)*100)+" %")

#SVM Confusion Matrix 
fig, ax = plt.subplots(figsize=(28,28))
sns.set(font_scale=1.8)  
sns.heatmap(cm,annot=True,square=True,cbar=False,xticklabels=labels,yticklabels=labels, linewidths=.5,fmt='g')

plt.title('Confusion Matrix')	


#SVM Classification Report 
print("Classification report")
print(clf_r)


#Model Accuracy = 63%


#------------------------------------------------------------------------------------------------------------------------



#Extreme Gradient Boosting (XGB) Classification

#XG Boost Classifier 
SVM_classifier = SVC(kernel = 'linear', random_state = 0)
SVM_classifier.fit(X_train, y_train)

#Pickling XGB Classifier
pickle_out = open("XG_classifier.pickle", "wb")
pickle.dump (XG_classifier, pickle_out)
pickle_out.close()

#importing XGB Classifier
pickle_in = open("XG_classifier.pickle", "rb")
XG_classifier = pickle.load(pickle_in)



#Evaluation of XGB
y_pred = RF_classifier.predict(X_test)
y_pred = le.inverse_transform(y_pred)
y_test=le.inverse_transform(y_test)
clf_r=classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

labels=le.inverse_transform(range(28))
print("Test Accuracy "+str(ac.round(4)*100)+" %")

#XGB Confusion Matrix 
fig, ax = plt.subplots(figsize=(28,28))
sns.set(font_scale=1.8)  
sns.heatmap(cm,annot=True,square=True,cbar=False,xticklabels=labels,yticklabels=labels, linewidths=.5,fmt='g')

plt.title('Confusion Matrix')	


#XGB Classification Report 
print("Classification report")
print(clf_r)


#Model Accuracy = 62%


#----------------------------------------------------------------------------------------------------------------------------------


#Hence, we choose Random Forest Classification for our task
#Apply the model over Test Data
dataset = pd.read_csv ("test.csv")
dataset['review_description'].apply(corpus)
cv = CountVectorizer(max_features=1000)
X = cv.transform(dataset['review_description'].values).toarray()
y = RF_classifier.predict(X)
y=le.inverse_transform(y)
dataset['predicted_variety']=y
dataset.to_csv("Predicted_Test_Data.csv", index = “False”)

