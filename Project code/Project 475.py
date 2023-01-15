# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 23:32:00 2023

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

df = pd.read_csv("Hotel Reservations.csv")

print(df.head())
print(df.info())

#%%

df.drop("Booking_ID", axis=1, inplace=True)
df.drop("arrival_year", axis=1, inplace=True)
df.drop("arrival_date", axis=1, inplace=True)

#%%

print(df.groupby("booking_status").describe())

sns.countplot(data = df, x = "booking_status")

#%%

f, ax = plt.subplots(figsize=(42, 20))
cmap = sns.light_palette('seagreen')
sns.heatmap(df.corr(), annot=True , cmap=cmap, vmax=1,vmin=-1, annot_kws={"size": 20})

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(binwidth=0.5, x="market_segment_type", hue="booking_status", data=df, stat="count", multiple="dodge")
plt.xticks(rotation=90)
#%%
sns.histplot(binwidth=0.5, x="room_type_reserved", hue="booking_status", data=df, stat="count", multiple="dodge")
plt.xticks(rotation=90)
#%%
sns.histplot(binwidth=0.5, x="type_of_meal_plan", hue="booking_status", data=df, stat="count", multiple="dodge")
plt.xticks(rotation=90)
#%%
sns.displot(data=df, x="lead_time", kind="kde")
sns.displot(data=df, x="avg_price_per_room", kind="kde")

#%%

print("\nNumber of duplicate rows =",list(df.duplicated()).count(True))
df=df.drop_duplicates()


#%%

from sklearn.preprocessing import LabelEncoder

class_encoder = LabelEncoder()
class_encoder.fit(df['booking_status'])
class_values = class_encoder.transform(df['booking_status'])

print("\nEncoding booking_status values:\n")
class_unique_values = df["booking_status"].unique()
encoded_unique_values = pd.DataFrame(class_values)[0].unique()
#print(encoded_unique_values)
for i in range(len(class_unique_values)):
    print(encoded_unique_values[i],"=",class_unique_values[i])
print("\n")

labels=df['booking_status']
df.drop("booking_status", axis=1, inplace=True)

df['booking_status']= class_values
print(df.head())
df.drop("booking_status", axis=1, inplace=True)


#%%

encoder=LabelEncoder()
encoder.fit(df['type_of_meal_plan'])
encoded_values=encoder.transform(df['type_of_meal_plan'])
df['type_of_meal_plan']=encoded_values

encoder.fit(df['room_type_reserved'])
encoded_values=encoder.transform(df['room_type_reserved'])
df['room_type_reserved']=encoded_values

encoder.fit(df['market_segment_type'])
encoded_values=encoder.transform(df['market_segment_type'])
df['market_segment_type']=encoded_values

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
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

# Define a series of parameters
params = dict(n_neighbors=[5,10,15,20,25])

# Create a random search cv object and fit it to the data
grid_search = GridSearchCV(knn, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, y)

# Print results
print("Best Accuracy: ",grid_search_results.best_score_)
print("Best Parameters: ",grid_search_results.best_params_)

list_CV_scr.append(grid_search_results.best_score_)

knn = KNeighborsClassifier(n_neighbors=grid_search_results.
                           best_params_['n_neighbors'])

knn.fit(X_train, y_train)

"""
k_folds = KFold(n_splits = 5)

scores = cross_val_score(knn, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("\nAverage CV Score: ", scores.mean())
list_CV_scr.append(grid_search_results.best_score_)
"""
print('\nTest accuracy: ',knn.score(X_test, y_test))
list_Test_acc_score.append(knn.score(X_test, y_test))

y_pred_test = knn.predict(X_test)
y_test_decoded = class_encoder.inverse_transform(y_test)
y_pred_test=class_encoder.inverse_transform(y_pred_test)

CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)


#%%

print("\n\t\t\t\t Adaboost Classifier\n")

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

adb = AdaBoostClassifier()

# Define a series of parameters
params = dict(n_estimators= [100, 250, 500], learning_rate = [0.01, 0.1, 1.0])

# Create a random search cv object and fit it to the data
grid_search = GridSearchCV(adb, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, y)

# Print results
print("Best Accuracy: ",grid_search_results.best_score_)
print("Best Parameters: ",grid_search_results.best_params_)

list_CV_scr.append(grid_search_results.best_score_)

adb = AdaBoostClassifier(n_estimators=grid_search_results.best_params_['n_estimators'],
                         learning_rate=grid_search_results.best_params_['learning_rate'])

adb.fit(X_train,y_train)

print('\nTest accuracy: ',adb.score(X_test, y_test))
list_Test_acc_score.append(adb.score(X_test, y_test))

y_pred_test = adb.predict(X_test)
y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)
#%%

print("\n\t\t\t\t SVM\n")

from sklearn import svm

svm1 = svm.SVC()

# Define a series of parameters
params = dict(C=[50,20,10,1],
              kernel=['linear', 'poly', 'rbf', 'sigmoid'])

# Create a random search cv object and fit it to the data
grid_search = GridSearchCV(svm1, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, y)

# Print results
print("Best Accuracy: ",grid_search_results.best_score_)
print("Best Parameters: ",grid_search_results.best_params_)

list_CV_scr.append(grid_search_results.best_score_)

svm = svm.SVC(C = grid_search_results.
                           best_params_['C'], kernel = grid_search_results.
                           best_params_['kernel'])

svm.fit(X_train, y_train)

print('\nTest accuracy: ',svm.score(X_test, y_test))
list_Test_acc_score.append(svm.score(X_test, y_test))

y_pred_test = svm.predict(X_test)
y_test_decoded = class_encoder.inverse_transform(y_test)
y_pred_test=class_encoder.inverse_transform(y_pred_test)

CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)

#%%

print("\n\t\t\t\t Voting Classifier\n")

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier

svm = svm.SVC(C = 50, kernel='rbf')
knn = KNN(n_neighbors=15)
dt = DecisionTreeClassifier()

# Define a series of parameters
params = dict(criterion=['gini', 'entropy', 'log_loss'])

grid_search = GridSearchCV(dt, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, y)

dt = DecisionTreeClassifier(criterion = grid_search_results.
                            best_params_['criterion'])


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
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)
#%%

print("\n\t\t\t\t RandomForest Classifier\n")

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

# Define a series of parameters
params = dict(n_estimators=[50,100,250,500], criterion=['gini', 'entropy', 'log_loss'])

# Create a random search cv object and fit it to the data
grid_search = GridSearchCV(rf, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, y)

# Print results
print("Best Accuracy: ",grid_search_results.best_score_)
print("Best Parameters: ",grid_search_results.best_params_)

list_CV_scr.append(grid_search_results.best_score_)

rf = RandomForestClassifier(n_estimators = grid_search_results.
                           best_params_['n_estimators'], criterion =
                           grid_search_results.best_params_['criterion'])

# Fit 'rf' to the training set
rf.fit(X_train, y_train)

print('\nTest accuracy: ',rf.score(X_test, y_test))
list_Test_acc_score.append(rf.score(X_test, y_test))

y_pred_test=rf.predict(X_test)
y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)


#%%

print("\n\t\t\t\t Deep Neural Network without batch processing\n")

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

def create_model_():
    # Instantiate a sequential model
    model = Sequential()
      
    # Add 3 dense layers of 128, 64 and 32 neurons each
    model.add(Dense(128, input_shape=(15,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
      
    # Add a dense layer with as many neurons as competitors
    model.add(Dense(2, activation='softmax'))
      
    # Compile your model using categorical_crossentropy loss
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#%%
from sklearn.preprocessing import OneHotEncoder

one_1 = y_train.reshape(len(y_train), 1)
one_2 = y_test.reshape(len(y_test), 1)
one_3 = y.reshape(len(y), 1)

### One hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_y_train = onehot_encoder.fit_transform(one_1)
onehot_encoded_y_test = onehot_encoder.fit_transform(one_2)
onehot_encoded_y = onehot_encoder.fit_transform(one_3)

#%%

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

model = KerasClassifier(build_fn=create_model_)

# Define a series of parameters
params = dict(epochs=[30,40,50,60,100])

# Create a random search cv object and fit it to the data
grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, onehot_encoded_y)

# Print results
print("Best Accuracy: ",grid_search_results.best_score_)
print("Best Parameters: ",grid_search_results.best_params_)

list_CV_scr.append(grid_search_results.best_score_)

#%%

model = create_model_()

print(model.summary())
# Fit model to the training data for 200 epochs
model.fit(X_train, onehot_encoded_y_train, 
           epochs=grid_search_results.best_params_['epochs'], verbose=0) 

#list_Test_DNN_acc_score_1=[]

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(X_test, onehot_encoded_y_test)

print('\nTest Accuracy:', accuracy[1])
list_Test_acc_score.append(accuracy[1])

y_pred_test=model.predict(X_test)
y_pred_test = onehot_encoder.inverse_transform(y_pred_test)
y_pred_test = y_pred_test.reshape(1,len(y_pred_test))
y_pred_test = y_pred_test.flatten()

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)


#%%

print("\n\t\t\t\t Deep Neural Network with batch processing\n")

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

def create_model():
# Instantiate a sequential model
    model2 = Sequential()
      
    # Add 3 dense layers of 128, 64 and 32 neurons each
    model2.add(Dense(128, input_shape=(15,), activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dense(64, activation='relu'))
    model2.add(Dense(32, activation='relu'))
      
    # Add a dense layer with as many neurons as competitors
    model2.add(Dense(2, activation='softmax'))
      
    # Compile model using categorical_crossentropy loss
    model2.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model2

#print(model2.summary())


#%%

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

model2 = KerasClassifier(build_fn=create_model)

# Define a series of parameters
params = dict(epochs=[30,40,50],
              batch_size=[32,64,128,413])

# Create a random search cv object and fit it to the data
grid_search = GridSearchCV(model2, params, cv=5, n_jobs=-1, verbose=0)
grid_search_results = grid_search.fit(X, onehot_encoded_y)

# Print results
print("Best Accuracy: ",grid_search_results.best_score_)
print("Best Parameters: ",grid_search_results.best_params_)

list_CV_scr.append(grid_search_results.best_score_)

#%%

model2 = create_model()

# Fit model to the training data for 200 epochs
model2.fit(X_train, onehot_encoded_y_train, 
           epochs=grid_search_results.best_params_['epochs'],
           batch_size=grid_search_results.best_params_['batch_size'], verbose=0)

# Evaluate model accuracy on the test data
accuracy = model2.evaluate(X_test, onehot_encoded_y_test)

#list_Test_DNN_acc_score_2=[]

# Print accuracy
print('\nTest Accuracy:', accuracy[1])
list_Test_acc_score.append(accuracy[1])

y_pred_test=model2.predict(X_test)
y_pred_test = onehot_encoder.inverse_transform(y_pred_test)
y_pred_test = y_pred_test.reshape(1,len(y_pred_test))
y_pred_test = y_pred_test.flatten()

y_pred_test=class_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test_decoded, y_pred_test, labels=class_unique_values)
print("\n2 unique target values: ",class_unique_values)
print("\nConfusion Matrix for 2 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=class_unique_values, yticklabels=class_unique_values)

#%%
import matplotlib.pyplot as plt
plt.figure("Figure 2")

X = ['K-nearest Neighbors','AdaBoost','Support Vector Machine','Voting Classifier','Random Forest','DNN without Batch', 'DNN with Batch']
X_axis = np.arange(len(X))
  
plt.bar(X_axis + 0.00, list_Test_acc_score, width = 0.25, label = 'Test Accuracy')
plt.bar(X_axis + 0.25, list_CV_scr, width = 0.25, label = 'Avg.CV Accuracy')

"""
X.append("DNN")
X_axis = np.arange(len(X))
Y_axis = np.arange(1)
plt.bar(Y_axis + 5, list_Test_DNN_acc_score_1, width = 0.25, color='#00CDCD', label = 'DNN without Batch processiong Test Accuracy')
plt.bar(Y_axis + 5.25, list_Test_DNN_acc_score_2, width = 0.25, color='#CD2626', label = 'DNN with Batch processiong Test Accuracy') 
"""

plt.xticks(X_axis+.11, X, rotation = 90)
plt.xlabel("Models",fontsize=12)
plt.ylabel("Scores",fontsize=12)
plt.title("Test vs Avg. CV Accuracy Comparison")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=11, title_fontsize=10)
plt.show()

#%%

print("\nModels' Test Accuracy Scores:\n")

for i in range(0,len(X)):
    print(X[i],":", list_Test_acc_score[i])

print("\nModels' Average Cross Validation Scores:\n")

for i in range(0,len(X)):
    print(X[i],":", list_CV_scr[i])
    