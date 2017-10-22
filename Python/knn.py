import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from dataProcessing import obtainXY , plot_confusion_matrix

neighs = [5, 7, 10, 12, 15, 20, 25, 30]
weighs = {'uniform','distance'}
algos ={'ball_tree','kd_tree','brute'}

n_Sel = 5
w_Sel = 'uniform'
a_Sel = 'auto'
acc_sel = 0

class_names = ['(1)','(2)','(3)','(4)','(5)']

x,y = obtainXY()

#Set the limits for the training and test data
dsSize = x.shape[0]
tsSize = int(dsSize*.8)

#shuffling of the data
x_sparse = coo_matrix(x)
x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)
x_sparse = x

#create training set
x_train = x_sparse[0:tsSize-1,:]
y_train = y[0:tsSize-1]

#create test set 
x_test = x_sparse[tsSize:dsSize,:]
y_test = y[tsSize:dsSize]

meanX = np.mean(x_train,axis=0)
stdX = np.std(x_train,axis=0)

count = 0

for entry in x_train:
	x_train[count] = (entry - meanX)/stdX
	count = count +1


count = 0

for entry in x_test:
	x_test[count] = (entry - meanX)/stdX
	count = count +1


# Select best parameters
for neigh in neighs:
	for weigh in weighs:
		for algo in algos:
			clf = neighbors.KNeighborsClassifier(n_neighbors=neigh, weights=weigh,algorithm=algo)
			clf = clf.fit(x_train, y_train)
			pridiction = clf.predict(x_test)
			score = accuracy_score(y_test, pridiction)*100
			print("Neighbours = "+str(neigh)+", Weighting : "+weigh+",Algorthim: "+algo+", Accuracy = "+str(score))
			if (score > acc_sel):
				acc_sel = score
				n_sel = neigh
				w_sel = weigh
				a_sel = algo

print("Best values are: Neighbours = "+str(n_sel)+", Weighting : "+w_sel+",Algorthim: "+a_sel+", Accuracy = "+str(acc_sel))
#training svm classifer with gaussian kernel
clf = neighbors.KNeighborsClassifier(n_neighbors=n_sel, weights=w_sel, algorithm=a_sel)
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

joblib.dump(clf, 'knnPrototypeC5.pkl') 

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pridiction)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')


# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')