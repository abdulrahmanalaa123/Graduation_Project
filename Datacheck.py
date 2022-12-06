from re import L
import pandas as pd
import json
import numpy as np
from random import sample
from operator import itemgetter
import math
def getter(x,index):
    if isinstance(x,tuple):
        return itemgetter(index)(x)
    else:
        return x

def splittuple(dat,index):
    return dat.applymap(lambda x:getter(x,index))

def encoder(data,colum):
    replacer = dict(zip(data[colum].unique(),range(len(data[colum].unique()))))
    print(replacer)
    temp = data[colum].replace(replacer)
    print(temp)
    return temp

def get(x):
    return itemgetter(-1)(x)

def sampler(dat):
    mask = dat["Stage"] == "STIMULUS"
    targets = dat.loc[mask,"Target"].unique()
    participants = list(dat.loc[mask,"Participant"].unique())
    trainval = int(0.7*len(participants))

    finaltest = np.zeros(shape=(1,6291))
    finaltrain = np.zeros(shape=(1,6291))


    for target in targets:
        mask2 = dat["Target"] == target
        trainpar = sample(participants,trainval)
        testpar = list(set(participants).symmetric_difference(set(trainpar)))
        masktrain = dat["Participant"].isin(trainpar)
        masktest = dat["Participant"].isin(testpar)
        train = dat.loc[mask & mask2 & masktrain,dat.columns[2:-12]].values
        test = dat.loc[mask & mask2 & masktest,dat.columns[2:-12]].values
        finaltrain = np.vstack((finaltrain,train))
        finaltest = np.vstack((finaltest,test))

    terminal = finaltrain.shape[1]-1
    finaltrain = np.delete(finaltrain,0,axis = 0)
    #np.random.shuffle(finaltrain)
    finaltest = np.delete(finaltest,0,axis = 0)
    #np.random.shuffle(finaltest)
    targettrain = np.array(list(map(get,finaltrain)))
    targettest = np.array(list(map(get,finaltest)))
    finaltrain = np.delete(finaltrain,terminal,axis = 1)
    finaltest = np.delete(finaltest,terminal,axis = 1)


    print("Final test after:"+str(finaltest.shape))

    return finaltrain,finaltest,targettrain,targettest



if __name__ == "__main__":
    data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
    list1 = [[12,23,251,21],[21321,12,41,2],[124,214,124,5]]
    
    print(list(map(get,list1)))
    X_train,X_test,train_target,test_target = sampler(data)
    print(len(X_train))
    print(len(X_test))
    print(len(train_target))
    print(len(test_target))
