'''
feature 1 (avgTime):calculates average time between moves
feature 2 (avgSpeed):calculates a value similar to average speed of user moves 
feature 3 (TimeSpent):calculates time duration user spends online
feature 4 (avgMoves):calculates avg number of by mouse movements per second
feature 5 (avgx):avg x coordinate of mouse pointer
feature 6 (avgy):avg y coordinate of mouse pointer
feature 7 (numberOfClicks):number of mouse clicks in the session
feature 8 (drags):number of mouse drags in the session
'''

import numpy as np
import glob
import os

#For now only 8 naive features have been extracted for now, more features to be added and the existing ones are to be improved.

# FileList=["session_0041905381","session_1060325796","session_3320405034",  "session_3826583375"  ,"session_6668463071","session_8961330453","session_9017095287"]
user_name = "user9"
training_FileList = glob.glob("/home/vision/anomaly_detections/dataset/Mouse-Dynamics-Challenge/training_files/" + user_name + "/*")
test_FileList = glob.glob("/home/vision/anomaly_detections/dataset/Mouse-Dynamics-Challenge/test_files/" + user_name + "/*")
#import ipdb; ipdb.set_trace()
labels = np.genfromtxt("/home/vision/anomaly_detections/dataset/Mouse-Dynamics-Challenge/public_labels.csv", dtype=None, delimiter=',')
file_name_col = labels[:, 0]
file_name_col = file_name_col.astype(np.str)


def extract_features(FileList):
	featureVector=[None] * 8

	for fileName in FileList:
		#import ipdb; ipdb.set_trace()
		# data = np.genfromtxt("user7/"+fileName, dtype=None, delimiter=',')
		data = np.genfromtxt(fileName, dtype=None, delimiter=',')
		data = data[1:]

		# split data array for every minute
		time_col = data[:, 1]
		num_minutes = int(float(time_col[-1])/300)
		split_at = time_col.astype(float).searchsorted(np.asarray([(i+1)*60 for i in range(num_minutes)]))
		data_list = np.split(data, split_at)

		for data in data_list:
			rows=data.shape[0]
			avgSpeed=0
			meanx=0
			meany=0
			numberOfClicks=0
			drags=0
			TimeSpent = avgMoves = avgTime = 0
			if rows > 1:
				TimeSpent=data[rows-1,1].astype(np.float)-data[0,1].astype(np.float)
				for i in range (1,rows):
					timeT=data[i,1].astype(np.float)-data[i-1,1].astype(np.float)
					avgSpeed=avgSpeed+((np.linalg.norm([data[i,4].astype(np.float) - data[i-1,4].astype(np.float), data[i,5].astype(np.float) - data[i-1,5].astype(np.float)]))/(timeT*rows+1))
					#to avoid NaN, 1 is added in the denominator
					meanx=  meanx+	((data[i,1].astype(np.float)-data[i-1,1].astype(np.float))*(data[i-1,4].astype(np.float)))/ TimeSpent
					meany=  meany+	((data[i,1].astype(np.float)-data[i-1,1].astype(np.float))*(data[i-1,5].astype(np.float)))/ TimeSpent
					if ((data[i-1,3].astype(np.str)!="Drag") and (data[i,3].astype(np.str)=="Drag")):
						drags=drags+1
					if(data[i-1,2]!=data[i,2]):
						numberOfClicks=numberOfClicks+1
						
				avgMoves=(rows/TimeSpent)  
				avgTime= (TimeSpent/rows)
			features=np.array([avgTime,avgSpeed,TimeSpent,avgMoves,meanx,meany,numberOfClicks,drags])
			featureVector=np.row_stack((featureVector,features))

	# import ipdb; ipdb.set_trace()

	featureVector=featureVector[1:,:]
	#print(featureVector.shape)

	import pandas as pd
	df=pd.DataFrame(data=featureVector,columns=["avgTime",'avgSpeed','TimeSpent','avgMoves','meanx','meany','numberOfClicks','drags'])

	from IPython.display import display, HTML
	#display(df)

	return df


training_features = extract_features(training_FileList)


from pyodds.utils.importAlgorithm import algorithm_selection
from pyodds.utils import utilities
from pyodds.utils.plotUtils import visualize_distribution_static,visualize_distribution_time_serie,visualize_outlierscore,visualize_distribution
#import ipdb; ipdb.set_trace()

clf = algorithm_selection('lof', 1, 0.3)
clf.fit(training_features)
#import pandas as pd

print("Extracting test_features")
test_features = []
#import ipdb; ipdb.set_trace()
for filepath in test_FileList:
	if os.path.basename(filepath) in file_name_col:
		label_index = np.where(file_name_col==os.path.basename(filepath))[0][0]
		gtruth = labels[label_index][1].astype(int)
		test_features = extract_features([filepath])
		prediction_result = clf.predict(test_features)
		outlierness_score = clf.decision_function(test_features)
		print(gtruth, prediction_result)
		# visualize_distribution(test_features, prediction_result, outlierness_score)

