import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from dataProcessing import obtainXY , plot_confusion_matrix



class_names = ['(1)','(2)','(3)','(4)','(5)']

# C_parm = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]
# G_parm = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]
# C_sel = 10
# G_sel = 0.3
# Acc_sel = 0

x,y = obtainXY()

#Set the limits for the training and test data
dsSize = x.shape[0]
# csSize = int(dsSize*.8)
tsSize = int(dsSize*.8)

#shuffling of the data
x_sparse = coo_matrix(x)
x, x_sparse, y = shuffle(x, x_sparse, y, random_state=1)
# x_sparse = x_sparse.toarray()
x_sparse = x

#create training set
x_train = x_sparse[0:tsSize-1,:]
y_train = y[0:tsSize-1]

# #create cross-validation set
# x_cross = x_sparse[tsSize:csSize-1,:]
# y_cross = y[tsSize:csSize-1]

#create test set 
x_test = x_sparse[tsSize:dsSize,:]
y_test = y[tsSize:dsSize]

meanX = np.mean(x_train,axis=0)
stdX = np.std(x_train,axis=0)

# np.savetxt('norm.txt', (meanX,stdX))

count = 0

for entry in x_train:
	x_train[count] = (entry - meanX)/stdX
	count = count +1


count = 0

for entry in x_test:
	x_test[count] = (entry - meanX)/stdX
	count = count +1

clf1 = joblib.load('svmPrototypeC5.pkl')
clf2 = joblib.load('knnPrototypeC5.pkl')
clf3 = joblib.load('mlpPrototypeC5.pkl')
pridiction = []

prodiction = np.array([clf1.predict(x_test),clf2.predict(x_test),clf3.predict(x_test)])
for ii in range(0, prodiction.shape[1]):
	temp = prodiction[:,ii]
	counts = np.bincount(temp)
	finalPre = np.argmax(counts)
	pridiction.append(finalPre) 
	if y_test[ii] != finalPre:
	# if not (temp[0] == temp[1] and temp[1] == temp[2]):
		print(temp)
		print(y_test[ii])

score = accuracy_score(y_test, pridiction)
print ("Test Set accuracy score: " + str(score*100))

# joblib.dump(clf, 'svmPrototypeC3.pkl') 

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pridiction)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')