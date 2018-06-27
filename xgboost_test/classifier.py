import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time




def process(row,df1,df2,aplist):
	
	print(row.name)
	# if(row.name<2000):
	# 	for ap,value in row.items() :
	# 		if(ap!='labels'):
	# 			loc = int(list(row['labels'])[3])
	# 			# print(value)
	# 			if(pd.isna(value)==False):
	# 				row[ap]=row[ap]-df1.iloc[loc-1][ap]

	# 	# print(yes)
	# else:
	# 	for ap,value in row.items() :
	# 		if(ap!='labels'):
	# 			loc = int(list(row['labels'])[3])
	# 			if(pd.isna(value)==False):
	# 				row[ap]=row[ap]-df2.iloc[loc-1][ap]

	# return row
	for ap,value in row.items():
		if(ap==aplist[0]):
			continue
		elif(ap!='labels'):
			if(pd.isna(value)==False):
				row[ap]=row[ap]-row[aplist[0]]
	return row




	

files1 = []
for i in range(1,5):
	filename = 'loc'+str(i)+'.csv'
	files1.append(filename)

files2 = []

for i in range(1,5):
	filename = 'l_loc'+str(i)+'.csv'
	files2.append(filename)






# print(files)
lines = []
labels = []
# print(os.listdir('.'))
for filename in os.listdir('.'):
	rowcount = 0
	# if('.csv' in filename and filename!='total.csv'):
	if(filename in files1 ) :
		file = open(filename,'r')
		# print(filename)
		for line in file:
			line = line.strip()
			line = line.replace("\"","")
			row  = line.split(',')
			lines.append(row)
			rowcount = rowcount+1
			# if rowcount==20:
			# 	break
		label = filename.replace('.csv','')
		labels += [label]*rowcount
		

for filename in os.listdir('.'):
	rowcount = 0
	# if('.csv' in filename and filename!='total.csv'):
	if(filename in files2) :
		file = open(filename,'r')
		# print(filename)
		for line in file:
			line = line.strip()
			line = line.replace("\"","")
			row  = line.split(',')
			lines.append(row)
			rowcount = rowcount+1
		label = filename.replace('l_','')
		
		label =	label.replace('.csv','')	
		
		# print(label)
		labels += [label]*rowcount


# print(labels)



aplist = []
for line in lines :
	# print(line)
	for reading in line:
		# print(reading)
		if(reading != ''):
			# x = reading.split(';')
			# print(x)
			bssid,strength = reading.split(';')
			if(bssid not in aplist):
				aplist.append(bssid)

# print(len(aplist))
aplist = aplist[:60]
df = pd.DataFrame(columns = aplist) #, dtype = np.dtype(['float']*len(aplist)))


df[aplist]=df[aplist].apply(pd.to_numeric)
i = 0 
for line in lines:
	df.loc[i] = [np.nan for n in range(len(aplist))]
	for reading in line :
		if(reading!=''):	
			bssid , strength = reading.split(';')
			if(bssid in df.columns):
				pd.to_numeric(df[bssid])
				df.loc[i,bssid] = float(strength)


	i=i+1	

# print(labels)



# print(df.info())
# print(df.head())

		
clf = XGBClassifier()


# df['labels'] = labels


# print(df.info())

# df1 = df[:2000].groupby('labels').mean()
# df2 = df[2000:].groupby('labels').mean()





# print(df1)
# print(df2)

# df1 = df[:2000].groupby('labels').count()
# df2 = df[2000:].groupby('labels').count()

# print(df1)
# print(df2)


# df = df.apply(lambda row: process(row,df1,df2,aplist) , axis =1)

# dfnew1 = df[:2000]
# dfnew2 = df[2000:]

# dfnew1 = dfnew1[dfnew1['labels']=='loc1']
# dfnew2 = dfnew2[dfnew2['labels']=='loc1']

# print(dfnew1[aplist[0]].head())
# print(dfnew2[aplist[0]].head())

# print(dfnew1.head())
# print(dfnew2.head())


# df.drop('labels',inplace=True,axis=1)
# df.drop(aplist[0],inplace=True,axis=1)


df.fillna(0,inplace=True)





X_train,X_test,Y_train,Y_test = train_test_split(df[:2000],labels[:2000],test_size = 0)


X_train1,X_test1,Y_train1,Y_test1 = train_test_split(df[2000:],labels[2000:],test_size = 0.5) 


print(type(X_train))


for f in X_train.columns: 
    if X_train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(X_train[f].values)) 
        X_train[f] = lbl.transform(list(X_train[f].values))

for f in X_test.columns: 
    if X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(X_test[f].values)) 
        X_test[f] = lbl.transform(list(X_test[f].values))



for f in X_train1.columns: 
    if X_train1[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(X_train1[f].values)) 
        X_train1[f] = lbl.transform(list(X_train1[f].values))

for f in X_test1.columns: 
    if X_test1[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(X_test1[f].values)) 
        X_test1[f] = lbl.transform(list(X_test1[f].values))



# print(df.head())

X_train = np.array(X_train) 
# X_test=np.array(X_test)
X_test1 = np.array(X_test1) 
X_train = X_train.astype(float) 
X_test1 = X_test1.astype(float)


# print(X_test)
print(len(Y_test1))

# print(len(X_train))

print(len(X_test1))

clf.fit(X_train,Y_train)



# print(clf.predict(X_test1))
# print(Y_test1)
# print(Y_test)
print(clf.score(X_test1,Y_test1))


