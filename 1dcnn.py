import tensorflow as tf
import pandas as pd
import numpy as np
from Datacheck import sampler,splittuple,encoder,evensampler
from sklearn.model_selection import GridSearchCV
import inspect
import concurrent.futures
import sys
import pickle
from sklearn.model_selection import RepeatedKFold
import time
def splitter(list,n):
    split = int(len(list)/n) 
    for i in range(0, len(list),split):
        print(i+split)
        yield list[i:i + split]
def permuter(list):
    combinations = []
    for i in range(len(list[0])):
        combinations.extend(grapher([],[list[0][i]],list[1:]))
    return combinations
def grapher(permutations,stable,rest):
    if len(rest) == 0:
        permutations.append(stable)
        return stable
    else:
        for i in range(len(rest[0])):
            #rest[0][i] in array for it to be able to concat to append without 
            #changing the original reference
            temp = stable+[rest[0][i]]
            #slicing the array past its end doesnt give an out of bound
            #it gives an empty array
            grapher(permutations,temp,rest[1:])
    return permutations

def tuning(X,Y,model,shape,vals,params = [],batch = 32,epoch = 50,cv = 5,repeat = 1):
    #gettign the parameters and base values of the function
    base = inspect.getfullargspec(eval(model))
    #getting the base values for the eval for it to be the local dict for eval
    basedict = dict(zip(base[0][1:],base[3])) 
    summary = []
    best = 0
    bestparam = []
    bestmodel = 0
    #the batches and params are put here because the comaprison of sets is put down there if it was only to batches and epochs
    #i'd save the processing done with the sets and use -3 index of the keys index but after one iteration it for some reason adds
    #a builtin element to the dictionary when used as a local identifier for the eval method
    if "batches" not in params:
        basedict["batches"] = batch
    if "epochs" not in params:
        basedict["epochs"] = epoch
    #this is put here as well to save the processing of identifying it several times inside the loop
    basedict[model]= eval(model)
    #the current fast optimization for the code if i remove the set compariosn and just use the -3 indexing to remove
    #batches,epochs,model from teh input str the thing is i dont know when and why does the builtin get added to the dict
    for elem in vals:
        #current values of each updated parameter
        updater = dict(zip(params,elem))
        basedict.update(updater)
        keys = list(basedict.keys())
        diff = 1 if '__builtins__' in keys else 0 
        inputstr = ",".join(f"{key}" for key in keys[:(-3-diff)])
        print(inputstr)
        kfolds = RepeatedKFold(n_splits = cv,n_repeats = repeat,random_state = 42)
        #gets the avg of the kfolds
        avggloballocal = 0
        #tries to get the bestinstance of all kfolds which doesnt really matter but helps you save the mdel
        #although the important thign are the parameters
        bestinstance = 0
        #used to compare accuracy of instances needs to be reinitialized everytime
        instance =0

        print(kfolds.get_n_splits(X))
        for train,test in kfolds.split(X):
            X_train ,X_test = X[train],X[test]
            train_target,test_target = Y[train],Y[test]
            mod = eval(f"{model}({shape},{inputstr})",basedict)

            mod.fit(X_train,train_target,validation_split = 0.2
                ,batch_size = basedict["batches"],epochs = basedict["epochs"]
                ,callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor = "val_loss",
                        patience = 5,
                        restore_best_weights=True
                    )
                ]
            )
            evaluation = mod.evaluate(X_test,test_target,verbose = 0)[1]
            #instance is 0 for every model to find max instance in model
            if evaluation > instance:
                bestinstance = mod
            #avggloballocal accuracy which will be avgd in the end
            avggloballocal += evaluation
        avggloballocal /= cv
        if avggloballocal >= best:
            best = avggloballocal
            bestparam = elem
            bestmodel = bestinstance
        string = ", ".join(f"{param}:{basedict[param]}" for param in params)
        print(string)
        summary.append(f"accuracy :{evaluation} params:{string}")
    summary.append("BEST MODEL:"+",".join(f"{param}:{val}" for param,val in zip(params,bestparam))+f",accuracy:{best}")
    pickle.dump(bestmodel,open('GraduationProject/Emognition/bestmodel.pkl', 'wb'))
    return summary


def cnn1d(shape,convlayers=2,layers = 2,unit = 100,func="relu",output=11,filter = 64,kernel = 3,initializer = "he_uniform",
dropout= 0.5,droplayers =1,lossfun = "sparse_categorical_crossentropy",optimizers = "Adam",metric = ["accuracy"]):
    model = tf.keras.Sequential()
    #kernel_initializer = "glorot_uniform" by default
    model.add(tf.keras.layers.Conv1D(filters = filter,kernel_initializer=initializer,kernel_size=kernel,activation=func,input_shape = shape))
    for _ in range(convlayers-1):
        model.add(tf.keras.layers.Conv1D(filters=filter, kernel_size=kernel, activation=func))
    for k in range(droplayers):
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(units = unit,activation=func))
    model.add(tf.keras.layers.Dense(output,activation="softmax"))
    model.compile(loss=lossfun, optimizer=optimizers, metrics= metric)
    return model

def lstmo(shape,lstmlayer = 4,layers = 1,unitnormal = 100,func="relu",output=11
,unitlstm = 50,dropout= 0.5,lossfun = "sparse_categorical_crossentropy",optimizers = "adam",metric = ["accuracy"]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=unitlstm,return_sequences=True,input_shape=shape))
    model.add(tf.keras.layers.Dropout(dropout))
    for _ in range(lstmlayer-1):
        model.add(tf.keras.layers.LSTM(units = unitlstm,return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Flatten())
    for _ in range(layers-1):
        model.add(tf.keras.layers.Dense(units = unitnormal,activation=func))
    model.add(tf.keras.layers.Dense(units = output,activation = "softmax"))
    model.compile(loss=lossfun, optimizer=optimizers, metrics= metric)
    return model
if __name__ == "__main__":
    data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
    #it is turned into an object i dont know why but it's turned into an inta nd type changed before
    data["Target"] = encoder(data,"Target")
    #mask so i dont pass it through another func
    data = data[data["Stage"] == "STIMULUS"]
    #get the hr values
    hrdat = splittuple(data,0)
    print(hrdat.dtypes)
    # get the target values
    #put in an np.array as int idk why but it used to work without it now it 
    #reads its type not as an int but as an object while every var inside is an instance of an int
    #same for hrdat
    hrtarget = np.asarray(hrdat.values[:,-13]).astype(int)
    #get hte X values
    hrdat = np.asarray(hrdat.values[:,2:-13]).astype(int)
    #code for including ppinterval is commented in case we dont need it acc 66%
    ppinterval = splittuple(data,1)
    #commenting to try cv
    #X_train,X_test,train_target,test_target = evensampler(hrdat)

    #put in case you need ppinterval
    #X_train2,X_test2,_,_ = evensampler(ppinterval)
    #X_train = np.dstack((X_train,X_train2))
    #X_test = np.dstack((X_test,X_test2))
    #X_train=X_train[:,:1300,:]
    #X_test =X_test[:,:1300,:]

    
    #X_train = np.asarray(X_train).astype("float32")
    #X_test = np.asarray(X_test3).astype("float32")
    params = ["droplayers","dropout","initializer"]
    ranges = [[1,2,3],[0.2,0.4,0.3,0.5,0.8],['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']]
    vals = permuter(ranges)

    result = tuning(hrdat,hrtarget,"cnn1d",(hrdat.shape[1],1),vals,params)
    with open('GraduationProject/Emognition/parameter_logs(cv).txt', 'w') as sys.stdout:
        print(list(result))

    #vals = list(splitter(vals,2))
    #print(vals)
    #print(len(vals))
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #   result = executor.map(lambda x:tuning(hrdat,hrtarget,"cnn1d",(hrdat.shape[1],1),x,params),vals)
