import numpy as np
import pandas as pd

def getData (DIR):
    df = pd.read_csv(DIR)
    df = np.array(df,np.float32)
    X = df[:,:-1]
    Y = df[:,-1]
    Y = Y.reshape((X.shape[0],1))
    return X, Y

def regulariseData (X):
    XSizeColumn = X.shape[0]
    XSizeRow = X.shape[1]
    XSum = X.sum(axis = 0)
    XMean = XSum/XSizeColumn
    XMax = X.max(axis = 0)
    XMin = X.min(axis = 0)
    XRange = XMax - XMin
    XReg = (X-XMean)/XRange
    df = pd.DataFrame(XReg)
    return df

def sigmoid (X) :
    sig = 1.0 / ( 1.0 + np.exp(-1.0*X) )
    return sig

def appendones (X):
    o = np.ones((X.shape[0],1),np.float32)
    o = np.array(o,np.float32)
    x = np.hstack((o,X))
    return X
