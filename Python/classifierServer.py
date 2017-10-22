import zerorpc
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from dataProcessing import getTreeData
from normVals import imageX

class HelloRPC(object):
    def classify(self, name):
    	x = imageX('../data/leaf.jpg')
    	print("Data Read In")

    	clf1 = joblib.load('svmPrototypeC5.pkl')
    	clf2 = joblib.load('knnPrototypeC5.pkl')
    	clf3 = joblib.load('mlpPrototypeC5.pkl')

    	pridiction = np.array([clf1.predict(x)[0],clf2.predict(x)[0],clf3.predict(x)[0]])
    	print(pridiction)
    	# finalPre = clf1.predict(x)[0]

    	counts = np.bincount(pridiction)
    	finalPre = np.argmax(counts)

    	treeJson = getTreeData(str(finalPre))
    	return treeJson

s = zerorpc.Server(HelloRPC())
s.bind("tcp://0.0.0.0:4242")
s.run()
