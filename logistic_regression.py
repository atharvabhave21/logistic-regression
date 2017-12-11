import utility as ut
import pandas as pd
import numpy as np

TRAIN_DIR = 'train.csv'
TEST_DIR = 'test.csv'
Num_Epoch = 2236
ALPHA = 3e-2
def getHypothesis(Theta,X):
    Z = np.matmul(X,Theta)
    Y_Predicted = ut.sigmoid(Z)
    return Y_Predicted

def costFunction(Y_Predicted,Y_Real):
    error = np.average(-Y_Real*(np.log(Y_Predicted))-(1-Y_Real)*np.log(1-Y_Predicted))
    return error

def optimiseTheta(X,Theta,Y_Real,Y_Predicted):
    X = np.transpose(X)
    e = Y_Real - Y_Predicted
    Theta += ALPHA*np.dot(X,e)
    return Theta

print ("Training Complete")

train_x,train_y = ut.getData(TRAIN_DIR)
train_x = np.array(train_x,np.float32)
train_y = np.array(train_y,np.float32)
train_x = ut.regulariseData(train_x)
train_x = ut.appendones(train_x)
theta = np.zeros((train_x.shape[1],1))
theta = np.array(theta,np.float32)

for i in range (Num_Epoch):
    hypothesis = getHypothesis(theta,train_x)
    error = costFunction(hypothesis,train_y)
    theta = optimiseTheta(train_x,theta,train_y,hypothesis)
    print ("Epoch = {} Error = {}".format(i,error))

test_x,test_y = ut.getData(TEST_DIR)
test_x = np.array(test_x,np.float32)
test_x = ut.regulariseData(test_x)
test_x = ut.appendones(test_x)
Y_Predicted = getHypothesis(theta,test_x)
correct = 0
for j in range (Y_Predicted.shape[0]):
    if test_y[j] == 1 and Y_Predicted[j] > 0.5 :
        correct+=1
    if test_y[j] == 0 and Y_Predicted[j] <= 0.5 :
        correct+=1
accuracy = 100.0*correct/test_y.shape[0]
print (accuracy)
