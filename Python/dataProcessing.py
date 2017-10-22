import numpy as np
import cv2
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import json
from skimage import feature
import itertools

sqlite_file = 'leavesDatabaseLast.sqlite'


class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")

		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		# hist /= (hist.sum() + eps)
		return hist

def obtainHu(img):
	huInvars = cv2.HuMoments(cv2.moments(img)).flatten() #Obtain hu moments from normalised moments in an array
	huInvars = -np.sign(huInvars)*np.log10(np.abs(huInvars))
	# huInvars /= huInvars.sum()
	return huInvars


def imgDisplay(img):
	# blur = cv2.GaussianBlur(img,(5,5),0)
	# (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()


def imgPreProcess(imgPath): #Obtains hu moments from image at path location
	img = cv2.imread(imgPath,0) #Load grey-scale image
	# img = cv2.bilateralFilter(image,5,75,75)
	# img = cv2.GaussianBlur(image,(5,5),0)

	# fd = open(imgPath, 'rb')
	# rows = 480
	# cols = 640
	# f = np.fromfile(fd, dtype=np.uint8,count=rows*cols)
	# img = f.reshape((rows, cols)) #notice row, column format
	# fd.close()

	hu = obtainHu(img)
	lbp = LocalBinaryPatterns(24, 8)
	hist = lbp.describe(img)

	return hu, hist

def arrayStringEncode(originalArray):
		encodedString = ""
		for item in originalArray:
			encodedString = encodedString+str(item)+','

		encodedString = encodedString[:-1]
		return encodedString

def loadPre(): #Load parameters for leaf images in database
	conn = sqlite3.connect(sqlite_file)
	df = pd.read_sql_query("select * from trainingData;", conn) #Obtain all leaf entries in leaves table
	hus = []
	lbps = []
	count = 0

	for index,row in df.iterrows(): #Iterate through all leaf entries
		hu,hist = imgPreProcess(row['colouredPath']) #Hu moment for leaf
		hus.append(arrayStringEncode(hu)) #Create matrix of hu moments
		lbps.append(arrayStringEncode(hist))
		# histString = ""
		# for item in hist:
		# 	histString = histString+str(item)+','

		# histString = histString[:-1]
		# lbps.append(histString)
		count = count +1
		print(count)
	
	print("Processed images")

	# Write each Hu moment into a seperate variable in the table in the database
	# df['Hu1'] = [item[0] for item in hus]
	# df['Hu2'] = [item[1] for item in hus]
	# df['Hu3'] = [item[2] for item in hus]
	# df['Hu4'] = [item[3] for item in hus]
	# df['Hu5'] = [item[4] for item in hus]
	# df['Hu6'] = [item[5] for item in hus]
	# df['Hu7'] = [item[6] for item in hus]

	df['Hu'] = hus
	df['Lbp'] = lbps
	df.to_sql("processed5", conn, if_exists="replace") #Write to database
	print("Wrote to database")

def stringArrayDecode(encodedString):
		recovArray = []

		histList = encodedString.split(',')
		# if len(histList) not in sizeLbp:
		# 	sizeLbp.append(len(histList))
		for item in histList:
			recovArray.append(float(item))

		return recovArray

def obtainXY():
	conn = sqlite3.connect(sqlite_file)
	df = pd.read_sql_query("select * from processed5;", conn)
	
	x1 = df['Hu'].tolist()
	x2 = df['Lbp'].tolist()

	count = 0
	while (count < len(x1)):
		huArray = stringArrayDecode(x1[count])
		x1[count] = huArray
		lbpHist = stringArrayDecode(x2[count])
		x2[count] = lbpHist
		count = count+1

	# count = 0

	# for lbpString in x2:
	# 	lbpHist = stringArrayDecode(lbpString)
	# 	x2[count] = lbpHist
	# 	count = count+1

	y = df['treeID'].tolist()#np.array(df['treeID'].tolist()).flatten('F')
	x = np.column_stack((x1,x2)) #Convert
	return (x,y)

def imageX(imgPath):
	meanX, stdX = normVals()
	hu,hist = imgPreProcess(imgPath)
	x= np.append(hu,hist)
	x = x.reshape((1, x.shape[0]))
	x = (x - meanX)/stdX
	return(x)

def getTreeData(treeId):
	conn = sqlite3.connect('leavesDatabase.sqlite')

	query = "SELECT treeID, treaName, leafPath, dataInformation FROM treeTable WHERE treeID = "+treeId

	dataFrame = pd.read_sql_query(query,conn)
	dt = dataFrame.set_index('treeID').T.to_dict()
	jsonResult = json.dumps(dt[int(treeId)])

	print(jsonResult)
	return jsonResult

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Greens):
	plt.figure()
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	#     print("Normalized confusion matrix")
	# else:
	#     print('Confusion matrix, without normalization')
	# print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label',fontweight='bold')
	plt.xlabel('Predicted label',fontweight='bold')

	plt.show()