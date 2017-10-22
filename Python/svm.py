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

C_parm = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]
G_parm = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]
C_sel = 10
G_sel = 0.3
Acc_sel = 0

x,y = obtainXY()

#Set the limits for the training and test data
dsSize = x.shape[0]
# csSize = int(dsSize*.8)
tsSize = int(dsSize*.8)

#shuffling of the data
x_sparse = coo_matrix(x)
x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)
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

np.savetxt('norm.txt', (meanX,stdX))

count = 0

for entry in x_train:
	x_train[count] = (entry - meanX)/stdX
	count = count +1


count = 0

for entry in x_test:
	x_test[count] = (entry - meanX)/stdX
	count = count +1

# clf1 = joblib.load('svmPrototypeC3.pkl')
# clf2 = joblib.load('knnPrototypeC.pkl')
# clf3 = joblib.load('mlpPrototypeC.pkl')
# pridiction = []

# prodiction = np.array([clf1.predict(x_test),clf2.predict(x_test),clf3.predict(x_test)])
# for ii in range(0, prodiction.shape[1]):
# 	temp = prodiction[:,ii]
# 	counts = np.bincount(temp)
# 	finalPre = np.argmax(counts)
# 	pridiction.append(finalPre) 
# 	if y_test[ii] != finalPre:
# 	# if not (temp[0] == temp[1] and temp[1] == temp[2]):
# 		print(temp)
# 		print(y_test[ii])


# Select best C and gamma values
for C in C_parm:
	for Gam in G_parm:
		clf = svm.SVC(gamma = Gam, C = C)
		clf = clf.fit(x_train, y_train)
		pridiction = clf.predict(x_test)
		score = accuracy_score(y_test, pridiction)*100
		print("C = "+str(C)+", Gamma = "+str(Gam)+", Accuracy = "+str(score))
		if (score > Acc_sel):
			Acc_sel = score
			C_sel = C
			G_sel = Gam

print("Best parameter values are: C= "+str(C_sel)+" Gamma: "+str(G_sel))

#training svm classifer with gaussian kernel
clf = svm.SVC(gamma = G_sel, C = C_sel)
# scores = cross_val_score(clf, x_test, y_test, cv=3)
# print(scores)
# scored = np.mean(scores)
# print(scored)

clf = clf.fit(x_train, y_train)
print ("Training completed")

#checking the classifier accuracy
pridiction = clf.predict(x_test)
score = accuracy_score(y_test, pridiction)
print ("Test Set accuracy score: " + str(score*100))

joblib.dump(clf, 'svmPrototypeC5.pkl') 

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pridiction)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')


# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')