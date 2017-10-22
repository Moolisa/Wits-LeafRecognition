import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from dataProcessing import obtainXY,imgPreProcess 

def imageX(imgPath):
	# meanX, stdX = normVals()
	meanX, stdX = np.loadtxt('norm.txt')
	hu,hist = imgPreProcess(imgPath)
	x= np.append(hu,hist)
	x = x.reshape((1, x.shape[0]))
	x = (x - meanX)/stdX
	return(x)

def normVals():
	x,y = obtainXY()
	dsSize = x.shape[0]
	tsSize = int(dsSize*.8)
	x_sparse = coo_matrix(x)
	x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)
	x_sparse = x
	x_train = x_sparse[0:tsSize-1,:]
	y_train = y[0:tsSize-1]
	meanX = np.mean(x_train,axis=0)
	stdX = np.std(x_train,axis=0)
	return (meanX,stdX)