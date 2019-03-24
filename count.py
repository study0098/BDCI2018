import pandas as pd
import random
filehead = 'E:/dataset/supplychain/'
baseline = pd.read_csv(filehead + 'submit95916.csv', encoding = 'utf8')
testfile = pd.read_csv(filehead + 'submit95916.csv', encoding = 'utf8')
count = 5000
select = random.sample(range(0,104510),count)
temp = pd.DataFrame(list(range(0,count)),index = select,columns=['useless'])
testset = pd.merge(temp, baseline, left_index=True,right_index=True, how ='left')
testset.rename(columns={'week1':'base_week1','week2':'base_week2','week3':'base_week3','week4':'base_week4','week5':'base_week5'},inplace = True)
testfile.rename(columns={'week1':'test_week1','week2':'test_week2','week3':'test_week3','week4':'test_week4','week5':'test_week5'},inplace = True)
final = pd.merge(testset, testfile,on='sku_id', how ='left')
import numpy as np
final['avg'] = 0
final['avg'] = final.apply(lambda x: (np.square(x['base_week1']-x['test_week1']) + np.square(x['base_week2']-x['test_week2'])
                           + np.square(x['base_week3']-x['test_week3']) + np.square(x['base_week4']-x['test_week4'])
                           + np.square(x['base_week5']-x['test_week5']))/5, axis = 1)
RSME = np.sqrt(final.avg.mean())
ans = 1/(1+RSME)
print('5000: '+str(ans))

count = 10000
select = random.sample(range(0,104510),count)
temp = pd.DataFrame(list(range(0,count)),index = select,columns=['useless'])
testset = pd.merge(temp, baseline, left_index=True,right_index=True, how ='left')
testset.rename(columns={'week1':'base_week1','week2':'base_week2','week3':'base_week3','week4':'base_week4','week5':'base_week5'},inplace = True)
testfile.rename(columns={'week1':'test_week1','week2':'test_week2','week3':'test_week3','week4':'test_week4','week5':'test_week5'},inplace = True)
final = pd.merge(testset, testfile,on='sku_id', how ='left')
import numpy as np
final['avg'] = 0
final['avg'] = final.apply(lambda x: (np.square(x['base_week1']-x['test_week1']) + np.square(x['base_week2']-x['test_week2'])
                           + np.square(x['base_week3']-x['test_week3']) + np.square(x['base_week4']-x['test_week4'])
                           + np.square(x['base_week5']-x['test_week5']))/5, axis = 1)
RSME = np.sqrt(final.avg.mean())
ans = 1/(1+RSME)
print('10000: '+str(ans))

count = 104510
select = random.sample(range(0,104510),count)
temp = pd.DataFrame(list(range(0,count)),index = select,columns=['useless'])
testset = pd.merge(temp, baseline, left_index=True,right_index=True, how ='left')
testset.rename(columns={'week1':'base_week1','week2':'base_week2','week3':'base_week3','week4':'base_week4','week5':'base_week5'},inplace = True)
testfile.rename(columns={'week1':'test_week1','week2':'test_week2','week3':'test_week3','week4':'test_week4','week5':'test_week5'},inplace = True)
final = pd.merge(testset, testfile,on='sku_id', how ='left')
import numpy as np
final['avg'] = 0
final['avg'] = final.apply(lambda x: (np.square(x['base_week1']-x['test_week1']) + np.square(x['base_week2']-x['test_week2'])
                           + np.square(x['base_week3']-x['test_week3']) + np.square(x['base_week4']-x['test_week4'])
                           + np.square(x['base_week5']-x['test_week5']))/5, axis = 1)
RSME = np.sqrt(final.avg.mean())
ans = 1/(1+RSME)
print('all: '+str(ans))