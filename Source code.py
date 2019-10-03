# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv("Master_Database.csv")
X = dataset.iloc[:,[2,3,4,5]].values
y = dataset.iloc[:, [0,1]].values


#Data Preprocessing

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X[:, 0]=le.fit_transform(X[:,0])
X[:, 1]=le.fit_transform(X[:,1])
X[:, 2]=le.fit_transform(X[:,2])
X[:, 3]=le.fit_transform(X[:,3])
y[:, 0]=le.fit_transform(y[:,0])
y[:, 1]=le.fit_transform(y[:,1])

onehotencoder=OneHotEncoder(categorical_features=[0,1,2,3])
X=onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#fitting Knearest neighbors to traioning set

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')
classifier.fit(X_train,y_train)




#predicting the set results

y_pred_knn=classifier.predict(X_test)

#Calculating cacuracy for Knn model

knn_acc=np.sum(np.not_equal(y_test, y_pred_knn))/float(y_test.size)
print(knn_acc)



#fitting decision tree

from sklearn.tree import DecisionTreeClassifier
classifier_dt=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_dt.fit(X_train, y_train)

#Predicting the Test set results

y_pred_dt = classifier_dt.predict(X_test)

#Calculating cacuracy for Decision tree model

dt_acc=np.sum(np.not_equal(y_test, y_pred_dt))/float(y_test.size)
print(dt_acc)


#fitting randomforest.

from sklearn.ensemble import RandomForestClassifier
classifier_rf= RandomForestClassifier(n_estimators=500,  criterion='entropy', random_state=0)
classifier_rf.fit(X_train, y_train)

#Predicting the Test set results

y_pred_rf = classifier_rf.predict(X_test)

#Calculating cacuracy for Random Forest model

rf_acc=np.sum(np.not_equal(y_test, y_pred_rf))/float(y_test.size)
print(rf_acc)







#Visualising the Comparitive analysis of performance of different algorithms. 

objects = ('K-Nearest Neighbours', 'Decision Tree', 'Random Forest')
y_pos = np.arange(len(objects))
performance = [knn_acc,dt_acc,rf_acc]
labels=[knn_acc,dt_acc,rf_acc]
plt.bar(y_pos, performance, align='center', alpha=0.8)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy in Percentage.')
plt.title('Different Classification Models')
for i in range (len(y_pos)):
    plt.text(x= y_pos[i]-0.3, y= performance[i]+1, s=labels[i], size=10)

 
plt.show()
