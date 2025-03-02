# importing libraries
import numpy as np

# importing dataset
from sklearn.datasets import load_digits
dataset = load_digits()

'''
# summarizing data - not needed in Spyder notebook
print(dataset.data)
print(dataset.target)
print(dataset.data.shape)
print(dataset.images.shape)
'''
dataImageLength = len(dataset.images)
#print(dataImageLength)


# Summarizing the data - optional
import matplotlib.pyplot as plt
'''for i in range(0, 15):  # we are just checking some random data images, can go up to 1796
    plt.gray()
    plt.matshow(dataset.images[i])
    plt.show()
print(dataset.images[1700])   # we will see the array of data for the particular image
'''

# Segregating the data
X = dataset.images.reshape((dataImageLength, -1)) # final part contains the output - so removed
y = dataset.target

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# importing model
from sklearn import svm
model = svm.SVC(kernel='linear') # will take default parameters values

# Training the model
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predicting the digit from the test data
n = 1005
result = model.predict(dataset.images[n].reshape(1,-1))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print(result)
print('\n')
plt.axis('off')
plt.title('%i' %result)
plt.show()

# for getting side by side comparison
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# trying other parameters
model1 = svm.SVC(kernel='rbf')
model2 = svm.SVC(gamma=0.001)
model3 = svm.SVC(gamma=0.001, C=0.1)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

print("Accuracy1:", accuracy_score(y_test, y_pred1))
print("Accuracy2:", accuracy_score(y_test, y_pred2))
print("Accuracy3:", accuracy_score(y_test, y_pred3))