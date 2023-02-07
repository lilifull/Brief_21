# Import
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# values import
df = pd.read_csv('car_truck_0.4.csv',index_col=0)

# Separate data in train and test
X_train, X_test, y_train, y_test = train_test_split(df.loc[:,df.columns!="label"], df.label, test_size=0.1, random_state=42)

# Standardize the labels
labelEncoder = preprocessing.LabelEncoder().fit(y_train)
y_train = labelEncoder.transform(y_train)
y_test = labelEncoder.transform(y_test)

# Standardize the data
scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model to be improved
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
model.fit(X_train, y_train)
base_accuracy_test = model.score(X_test, y_test)
base_accuracy_train = model.score(X_train, y_train)

# best model
RFC = RandomForestClassifier(max_leaf_nodes = 15000, n_jobs = -1, random_state=20,bootstrap=True, n_estimators= 300, criterion='gini')
result = RFC.fit(X_train, y_train)
accuracy_test = RFC.score(X_test, y_test)
accuracy_train = RFC.score(X_train, y_train)
print('----------------------------------------------------------------------------------------------------------------')
print(f"---->   Précision entrainement de base : {base_accuracy_train}, précision test de base : {base_accuracy_test}")
print(f"---->   Précision entrainement amélioré : {accuracy_train}, précision test amélioré: {accuracy_test}")
print('=================================================================================================================')
plot_confusion_matrix(RFC, X_test, y_test) 
plt.show()



