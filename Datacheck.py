from re import L
import pandas as pd
import json
import numpy as np
import random
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

    random.seed(123)
    for target in targets:
        mask2 = dat["Target"] == target
        trainpar = random.sample(participants,trainval)
        print(trainpar)
        testpar = list(set(participants).symmetric_difference(set(trainpar)))
        masktrain = dat["Participant"].isin(trainpar)
        masktest = dat["Participant"].isin(testpar)
        train = dat.loc[mask & mask2 & masktrain,dat.columns[2:-12]].values
        test = dat.loc[mask & mask2 & masktest,dat.columns[2:-12]].values
        finaltrain = np.vstack((finaltrain,train))
        finaltest = np.vstack((finaltest,test))


    finaltrain = finaltrain[1:,:]
    #np.random.shuffle(finaltrain)
    finaltest = finaltest[1:,:]
    #np.random.shuffle(finaltest)
    targettrain = finaltrain[:,-1:]
    targettest = finaltest[:,-1:]
    finaltrain = finaltrain[:,:6290]
    finaltest = finaltest[:,:6290]



    return finaltrain,finaltest,targettrain,targettest

def evensampler(data):
    mask = data["Stage"] == "STIMULUS"
    participants = data["Participant"].unique()
    trainval = int(0.7*len(participants))
    trainpar = participants[:trainval]
    testpar = participants[trainval:]
    masktrain = data["Participant"].isin(trainpar)
    masktest = data["Participant"].isin(trainpar)
    train = data.loc[mask&masktrain,data.columns[2:-13]].values
    traintarget = data.loc[mask&masktest,"Target"].values
    test = data.loc[mask&masktrain,data.columns[2:-13]].values
    testtarget = data.loc[mask&masktest,"Target"].values

    return train,test,traintarget,testtarget
if __name__ == "__main__":
    data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
    list1 = np.array([[12,23,251,21],[21321,12,41,2],[124,214,124,5]])
    
    list1 = list1[1:,:3]
    print(list1)
    X_train,X_test,train_target,test_target = sampler(data)
    print(X_train.shape)
    print(X_test.shape)
    print(len(train_target))
    print(len(test_target))
