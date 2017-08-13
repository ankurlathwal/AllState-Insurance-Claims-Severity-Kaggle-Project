#CS 514 Applied Artificial Intelligence
#Project 5
#AllState Insurance Claims Severity Problem


# Necessary imports
import numpy as np
from sklearn import linear_model
from numpy import genfromtxt
from sklearn.metrics import mean_absolute_error

# Training data subset (file included in the project folder)
# Reading csv file into a numpy array
train = genfromtxt('train_subset.csv', delimiter = ',')

# Testing data subset (file included in the project folder)
test = genfromtxt('test_subset.csv', delimiter = ',')

# divide training subset into data and target
train_x= train[:,0:130]
train_y = train[:,131]

# testing data
test_x = train[:,0:130]

# Applying linear regression model
clf = linear_model.LinearRegression()

# training linear regression using training subset data
# feeding learning data to the algorithm
clf.fit(train_x,train_y)

# printing prediction on testing data
# print(clf.predict(test_x))

# Mean Absolute Error (MAE)
mae  = mean_absolute_error(clf.predict(test_x), train_y)
print("Mean Absolute Error (MAE): %.4f" % mae)

# storing result (prediction) in an array
result = clf.predict(test_x)
# IDs of the submission
id = test[:,0]

# combining IDs and loss predictions
output = np.vstack((id, result))

# save output to a file named submission.csv, which will be created and saved in the project folder after code is executed
np.savetxt(
    'submission.csv',           # file name
    output,                 # array to save
    fmt='%.2f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    comments='# ',          # character to use for comments
    )

