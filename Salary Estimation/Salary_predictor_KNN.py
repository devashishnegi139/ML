# importing libraries
import pandas as pd
import numpy as np

# importing data
dataset = pd.read_csv('salary.csv')

# summarizing data
#print(dataset.shape)
#print(dataset.head(5))

# mapping data
income_set = set(dataset['income']) # to get the number of different classes
dataset['income'] = dataset['income'].map({'<=50K':0, '>50K':1}).astype(int)
#print(dataset.head)
# now we have a clean dataset - only numbers

# segregating the dataset
X = dataset.iloc[:, :-1].values # we will get just values, not the column names
y = dataset.iloc[:, -1].values

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
# if done before splitting then model will get an insight from the test dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# importing KNN model
from sklearn.neighbors import KNeighborsClassifier as KNN

'''# Finding the best k-value
error = []
import matplotlib.pyplot as plt

# calculating error values for k in range 1-40
for i in range(1, 40):
    model = KNN(n_neighbors=i) # setting model for corresponding i values
    model.fit(X_train, y_train)
    y_pred_i = model.predict(X_test)
    error.append(np.mean(y_pred_i != y_test))
# y_pred_i != y_test, creates a boolean array of 0 and 1
# np.mean calculated it's mean

# plotting the graph of it
plt.figure(figsize=(12,6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("Error Rate of K-Value")
plt.xlabel('K Value')
plt.ylabel('Mean Error')'''

# Training our model based on new k value
model = KNN(n_neighbors=16, metric='minkowski', p=2) # p=2 for Euclidean Distance
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# the above can be used to see the comparison side by side, but we can use metrices

# Evaluating
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix:\n", cm)

'''# Validating the model
age = int(input("Enter New Employee's Age: "))
edu = int(input("Enter New Employee's Education: "))
cg = int(input("Enter New Employee's Capital Gain: "))
wh = int(input("Enter New Employee's Hours per week: "))
newEmp = [[age, edu, cg, wh]]
result = model.predict(sc.transform(newEmp))
if result==1:
    print("The employee may have received a salary of 50K or more.")
else:
    print("The employee may not have received a salary of 50K or more.")'''