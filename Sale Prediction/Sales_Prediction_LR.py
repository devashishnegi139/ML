# importing Libraries
import pandas as pd

# importing dataset
dataset = pd.read_csv('DigitalAd_dataset.csv')
# we can also save data in google drive and import it

# summarizing dataset
print(dataset.shape) # to get the number of rows and columns
print(dataset.head(5)) # to get first 5 values

# segregating dataset
X = dataset.iloc[:, :-1].values # will get all rows and (column-lastColumn)
y = dataset.iloc[:, -1].values # will get all rows and just last column

# Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler # standardScalar is also a method
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # 
# y doesn't need feature scaling - it's just 0 and 1

# importing the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)

# Training the model
model.fit(X_train, y_train)

# Testing the model
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix:\n", cm)

# Validating the model
age = int(input("Enter new Customers' Age: "))
salary = int(input("Enter new Customer's Salary: "))
newCustomer = [[age,salary]]
newCustomer = sc.transform(newCustomer)
result = model.predict(newCustomer)
if(result == 1):
    print("Customer will Buy!")
else:
    print("Customer will not Buy!")
    
