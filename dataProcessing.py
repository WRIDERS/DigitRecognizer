import csv;
from sklearn import ensemble;
from sklearn import linear_model;

class readData():
	
	dataHeader=[];
	TrainDataX=[];
	TrainDataY=[];
	TestDataX=[];
	PredictDataY=[];
	
	def ListtoInt(self,L):
		temp=[];
		for x in L:
			temp+=[int(x)];
		return temp;
	
	
	def readTrainData(self):
		with open ('DATA/train.csv','rb') as csvfile:
			csvreader=csv.reader(csvfile,delimiter=',',quotechar='"');
			self.dataHeader=csvreader.next();
			for rows in csvreader:
				self.TrainDataY+=[int(rows[0])];
				self.TrainDataX+=[self.ListtoInt(rows[1:])];
		with open ('DATA/test.csv','rb') as csvfile:
			csvreader=csv.reader(csvfile,delimiter=',',quotechar='"');
			csvreader.next();
			for rows in csvreader:
				self.TestDataX+=[self.ListtoInt(rows)];
		print('read Done');	
				
	def trainModel(self):
		Classifier=ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=1, min_samples_leaf=1, min_density=0.10000000000000001, max_features='auto', bootstrap=True, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0);		
		#Classifier=linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=100000.0, fit_intercept=True, intercept_scaling=1, class_weight=None);
		print('Training Complete');
		Classifier.fit(self.TrainDataX,self.TrainDataY);		
		self.PredictDataY=Classifier.predict(self.TestDataX);
		with open('DATA/out_v1.csv','wb') as csvfile:
			csvwriter=csv.writer(csvfile,delimiter=',',quotechar='"');
			csvwriter.writerow(self.dataHeader[0]);
			for x in self.PredictDataY:
				csvwriter.writerow([x]);
		
		

		

