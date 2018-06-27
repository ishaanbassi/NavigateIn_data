import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import csv 

maindict = {}
for i in range(1,8):
	print('opening file - Location '+str(i))
	f = open('Location '+str(i)+'.csv','r')
	l1 = csv.reader(f)
	dictionary={}
	for reading in l1:
		for ap in reading:
			# print(ap)
			if(ap!=''):
				id,strength=ap.split(';')
				
				if(id not in dictionary):
					lis=[]
					lis.append(int(strength))
					dictionary[id]=lis
				else:
					lis=dictionary[id]
					lis.append(int(strength))
					dictionary[id] = lis
		# print(dictionary)
	for id in dictionary:
		lis = dictionary[id]
		# print(lis)
		mean =  float(sum(lis))/len(lis)
		if(id not in maindict):
			items = []
			items.append(mean)
			maindict[id] = items
		else:
			items =  maindict[id]
			items.append(mean)
			maindict[id]=items

print(maindict)
count = 1
for ap in maindict:
	if(len(maindict[ap])==7):
		xvalues = np.array([1,2,3,4,5,6,7])
		yvalues = np.array(maindict[ap])
		plt.plot(xvalues,yvalues)
		plt.savefig('plt'+str(count)+'.pdf')
		plt.gcf().clear()
		count = count+1


# print(dictionary.values())


