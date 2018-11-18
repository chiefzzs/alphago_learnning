import time
import pickle
import os

path = os.path.dirname(os.path.realpath(__file__))
path ="F:/learnning/ai/data"

def backupSave(data,fname):
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    fullName=path+"/trainData/"+fname+"_"+now+".data"
    f= open(fullName, 'wb')
    pickle.dump(data, f)
    f.close()
    return fullName


def pushHumanSave(data,fname):
    fullName=path+"/selfCreateData/"+fname+"_"+".data"
    f= open(fullName, 'wb')
    pickle.dump(data, f)
    f.close()
    return fullName

def backupLoad(fname):
    f= open(fname, 'rb')
    data = pickle.load(f)
    f.close()
    return data


#s = [1, 2, 3, 4, 5]
#fName = save(s,"test")
#print(fName)
#print(load(fName))
