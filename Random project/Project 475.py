# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 19:02:44 2023

@author: User
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("SD3_data.csv")

print(df.head())
print(df.info())

#%%
df.drop("country", axis=1, inplace=True)
df.drop("source", axis=1, inplace=True)

#%%

import statistics as st

labels=[]

for i in range(0,18192):
    m = st.mean(list(df.loc[i, 'M1':'M9']))
    n = st.mean(list(df.loc[i, 'N1':'N9']))
    p = st.mean(list(df.loc[i, 'P1':'P9']))
    
    if((3.68-n)>(3.86-m)<(3.40-p)):
        labels.append('M')
    elif((3.86-m)>(3.68-n)<(3.40-p)):
        labels.append('N')
    else:
        labels.append('P')


df['class']= labels

print(df.head())


#%%

print(df.groupby("class").describe())

sns.countplot(data = df, x = "class")

#%%

f, ax = plt.subplots(figsize=(42, 20))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(), annot=True , cmap=cmap, vmax=1,vmin=-1, annot_kws={"size": 20})

#%%

print("\nNumber of duplicate rows =",list(df.duplicated()).count(True))
df=df.drop_duplicates()

#%%

from sklearn.preprocessing import LabelEncoder

class_encoder = LabelEncoder()
class_encoder.fit(df['class'])
class_values = class_encoder.transform(df['class'])

print("\nEncoding class values:\n")
class_unique_values = df["class"].unique()
encoded_unique_values = pd.DataFrame(class_values)[0].unique()
#print(encoded_unique_values)
for i in range(len(class_unique_values)):
    print(encoded_unique_values[i],"=",class_unique_values[i])
print("\n")

labels=df['class']
df.drop("class", axis=1, inplace=True)

df['class']= class_values
print(df.head())
df.drop("class", axis=1, inplace=True)

#%%

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df = ss.fit_transform(df)
print(df)
print("\n")

#%%

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.figure("New Figure")

pca = PCA()


pca.fit(df)

df= pca.transform(df)

features = range(pca.n_components_)


plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

#%%

list_CV_scr=[]
list_Test_acc_score=[]
#%%

X=df
y=class_values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

#%%

print("\n\t\t\t\t K-Neighbors Classifier\n")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred_test = knn.predict(X_test)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(knn, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())
list_CV_scr.append(scores.mean())

print('\nTest accuracy: ',knn.score(X_test, y_test))
list_Test_acc_score.append(knn.score(X_test, y_test))

y_test_decoded = class_encoder.inverse_transform(y_test)
y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)

#%%

print("\n\t\t\t\t Adaboost Classifier\n")

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(max_depth = 10, random_state=1)

adb = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

adb.fit(X_train,y_train)

y_pred_test = adb.predict(X_test)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(adb, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())
list_CV_scr.append(scores.mean())

print('\nTest accuracy: ',adb.score(X_test, y_test))
list_Test_acc_score.append(adb.score(X_test, y_test))

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)
#%%

print("\n\t\t\t\t SVM\n")

from sklearn import svm

svm = svm.SVC(C=0.1, kernel='poly')

svm.fit(X_train, y_train)

y_pred_test = svm.predict(X_test)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(svm, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())
list_CV_scr.append(scores.mean())

print('\nTest accuracy: ',svm.score(X_test, y_test))
list_Test_acc_score.append(svm.score(X_test, y_test))

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)

#%%

"""print("\n\t\t\t\t Logistic Regression\n")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.0005)

lr.fit(X_train,y_train)

y_pred_test = lr.predict(X_test)

k_folds = KFold(n_splits = 5)
scores = cross_val_score(lr, X_train, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())

print('\nTest accuracy: ',lr.score(X_test, y_test))

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)"""

#%%

print("\n\t\t\t\t Voting Classifier\n")

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import VotingClassifier

svm = svm.SVC(C=0.5, kernel='poly', random_state=1)
knn = KNN(n_neighbors=7)
dt = DecisionTreeClassifier(max_depth = 12, random_state=1)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Support Vector Machine',svm),('K Nearest Neighbours', knn),('Classification Tree', dt)]

"""
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('\n{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
"""

    
vc = VotingClassifier(estimators=classifiers)

vc.fit(X_train, y_train)

y_pred_test = vc.predict(X_test)

k_folds = KFold(n_splits = 5)
scores = cross_val_score(vc, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())
list_CV_scr.append(scores.mean())

print('\nTest accuracy: ',vc.score(X_test, y_test))
list_Test_acc_score.append(vc.score(X_test, y_test))

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)
#%%

print("\n\t\t\t\t RandomForest Classifier\n")

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=600, min_samples_split=0.0005, random_state=1)

# Fit 'rf' to the training set
rf.fit(X_train, y_train)

y_pred_test=rf.predict(X_test)

k_folds = KFold(n_splits = 5)
scores = cross_val_score(rf, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())
list_CV_scr.append(scores.mean())

print('\nTest accuracy: ',rf.score(X_test, y_test))
list_Test_acc_score.append(rf.score(X_test, y_test))

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)


#%%

print("\n\t\t\t\t Deep Neural Network without batch processing\n")

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(27,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(3, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#%%
from sklearn.preprocessing import OneHotEncoder

one_1 = y_train.reshape(len(y_train), 1)
one_2 = y_test.reshape(len(y_test), 1)
one_3 = y.reshape(len(y), 1)

### One hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_y_train = onehot_encoder.fit_transform(one_1)
onehot_encoded_y_test = onehot_encoder.fit_transform(one_2)
#onehot_encoded_y = onehot_encoder.fit_transform(one_3)


model.fit(X_train, onehot_encoded_y_train, epochs=1)

list_Test_DNN_acc_score_1=[]

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(X_test, onehot_encoded_y_test)

print('\nTest Accuracy:', accuracy[1])
list_Test_DNN_acc_score_1.append(accuracy[1])

y_pred_test=model.predict(X_test)
y_pred_test = onehot_encoder.inverse_transform(y_pred_test)
y_pred_test = y_pred_test.reshape(1,len(y_pred_test))
y_pred_test = y_pred_test.flatten()

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)


#%%

print("\n\t\t\t\t Deep Neural Network with batch processing\n")

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Instantiate a sequential model
model2 = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model2.add(Dense(128, input_shape=(27,), activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model2.add(Dense(3, activation='softmax'))
  
# Compile model using categorical_crossentropy loss
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%


# Fit model to the training data for 200 epochs
model2.fit(X_train, onehot_encoded_y_train, epochs=1, batch_size=290) # 50 Batches

# Evaluate model accuracy on the test data
accuracy = model2.evaluate(X_test, onehot_encoded_y_test)

list_Test_DNN_acc_score_2=[]

# Print accuracy
print('\nTest Accuracy:', accuracy[1])
list_Test_DNN_acc_score_2.append(accuracy[1])

y_pred_test=model2.predict(X_test)
y_pred_test = onehot_encoder.inverse_transform(y_pred_test)
y_pred_test = y_pred_test.reshape(1,len(y_pred_test))
y_pred_test = y_pred_test.flatten()

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n3 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 3 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)

#%%
import matplotlib.pyplot as plt
plt.figure("Figure 2")

X = ['KNN','ADB','SVM','VC','RF']
X_axis = np.arange(len(X))
  
plt.bar(X_axis + 0.00, list_Test_acc_score, width = 0.25, label = 'Test Accuracy')
plt.bar(X_axis + 0.25, list_CV_scr, width = 0.25, label = 'Avg.CV Accuracy')

X.append("DNN")
X_axis = np.arange(len(X))
Y_axis = np.arange(1)
plt.bar(Y_axis + 5, list_Test_DNN_acc_score_1, width = 0.25, color='#00CDCD', label = 'DNN without Batch processiong Test Accuracy')
plt.bar(Y_axis + 5.25, list_Test_DNN_acc_score_2, width = 0.25, color='#CD2626', label = 'DNN with Batch processiong Test Accuracy') 


plt.xticks(X_axis+.11, X, )
plt.xlabel("Models",fontsize=12)
plt.ylabel("Scores",fontsize=12)
plt.title("Test vs Avg. CV Accuracy Comparison")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=11, title_fontsize=10)
plt.show()
