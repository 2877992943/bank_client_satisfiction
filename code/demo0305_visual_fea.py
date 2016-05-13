#!/usr/bin/env python3
# encoding=utf-8





dataPath='/home/yr/search_relevant/'

import time,pickle
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re
#import enchant
import random
random.seed(2016)


from sklearn.ensemble import RandomForestClassifier

import pylab as plt



 


def save2pickle(c,name):
	write_file=open(dataPath+str(name),'wb')
	pickle.dump(c,write_file,-1)
	write_file.close()

def load_pickle(path_i):
	f=open(dataPath+path_i,'rb')
	data=pickle.load(f)
	f.close()
	return data




 
		



training = pd.read_csv("../input/train.csv", index_col=0)#[:3,:]
test = pd.read_csv("../input/test.csv", index_col=0)#[:3,:]

print(training.shape,training.values.shape)#[7w,300]
print(test.shape)

print ('train',training.columns)
X = training.iloc[:,:-1]
y = training.TARGET

"""
print ( y.value_counts()/float(y.size)	)


# ratio of nonzero elements
print (X.apply(lambda x:x[x!=0].size).sum() / float(np.prod(training.shape))  ) 
#apply series or dataFrame
#axis=0 column
#axis=1 row


print (test.apply(lambda x:x[x!=0].size).sum() / float(np.prod(test.shape)) )

print (X.dtypes.value_counts() )


#print ( sum([re.sub("\d+", "", s).split("_") for s in X.columns],[])  )
  

name_component = pd.Series(sum([re.sub("\d+", "", s).split("_") for s in X.columns], []))
name_component.replace("", "_0", inplace=True)
name_component.value_counts()
"""

nuniques_train = X.apply(lambda x:x.nunique());#print (nuniques_train.shape,nuniques_train)#[300,]dim columns
nuniques_test = test.apply(lambda x:x.nunique());#print (nuniques_test.shape)
"""
var3                               199
var15                              100
imp_ent_var16_ult1                 598
imp_op_var39_comer_ult1           7593
...
"""

 
no_variation_train = nuniques_train[nuniques_train==1].index;
no_variation_test = nuniques_train[nuniques_test==1].index

print(no_variation_train.size, no_variation_test.size)
print(no_variation_train, no_variation_test)
#print (nuniques_train[no_variation_test],nuniques_test[no_variation_train])
print('\nTrain[no variation in test]\n#unique cnt\n',nuniques_train[no_variation_test].value_counts())
print('\nTest[no variation in train]\n#unique cnt\n', nuniques_test[no_variation_train].value_counts())

 
##remove columns where all values==1
X, test = [df.drop(no_variation_train, axis=1) for df in [X, test]]
nuniques_train, nuniques_test = [s.drop(no_variation_train) for s in [nuniques_train, nuniques_test]]


ax = nuniques_train[nuniques_train>200].hist(bins=200, figsize=(10, 7))
ax.set_xlabel("#uniques")
ax.set_title("Histogram of #uniques (>200)")
plt.show()




print (nuniques_train[nuniques_train<200].size)
print (nuniques_train[nuniques_train>=200].size)



#####
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X,y)
feat_imp=pd.Series(rf.feature_importances_,index=X.columns)
feat_imp.sort_values(inplace=True)
ax=feat_imp.tail(20).plot(kind='barh',figsize=(10,7),title='feature importance')





 
	



	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



