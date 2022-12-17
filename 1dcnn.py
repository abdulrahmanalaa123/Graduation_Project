import tensorflow as tf
import pandas as pd
import numpy as np
from Datacheck import sampler,splittuple,encoder,evensampler
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
import inspect
import concurrent.futures
import sys
import pickle
def splitter(list,n):
    for i in range(0, len(list), n):
        yield list[i:i + n]
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

def tuning(X_train,train_target,X_test,test_target,model,shape,vals,params = [],batch = 32,epoch = 50):
    base = inspect.getfullargspec(eval(model))
    print(len(vals))
    #getting the base values for the eval for it to be the local dict for eval
    basedict = dict(zip(base[0][1:],base[3])) 
    summary = []
    best = 0
    bestparam = []
    bestmodel = 0
    for elem in vals:
        #current values of each updated parameter
        updater = dict(zip(params,elem))
        basedict.update(updater)
        basedict[model]= eval(model)
        keys = list(basedict.keys())
        print(keys)
        diff = -len(set(keys)&set(["batches","epochs",model,'__builtins__']))
        if diff < 0:
            inputstr = ",".join(f"{key}" for key in keys[:diff])
        else:
            inputstr = ",".join(f"{key}" for key in keys[:])
        print(inputstr)
        mod = eval(f"{model}({shape},{inputstr})",basedict)
        
        if "batches" not in params:
            basedict["batches"] = batch
        if "epochs" not in params:
            basedict["epochs"] = epoch

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

        if evaluation >= best:
            best = evaluation
            bestparam = elem
            bestmodel = mod
        string = ", ".join(f"{param}:{basedict[param]}" for param in params)
        print(string)
        summary.append(f"accuracy :{eval} params:{string}")
    summary.append(f"{param}:{val}" for param,val in zip(params,bestparam))
    pickle.dumps(bestmodel,"GraduationProject/Emognition/bestmodel from tuning")
    return summary


def cnn1d(shape,convlayers=2,layers = 2,unit = 100,func="relu",output=11,filter = 64,kernel = 3,
dropout= 0.5,droplayers =1,lossfun = "sparse_categorical_crossentropy",optimizers = "adam",metric = ["accuracy"]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters = filter,kernel_size=kernel,activation=func,input_shape = shape))
    for _ in range(convlayers-1):
        model.add(tf.keras.layers.Conv1D(filters=filter, kernel_size=kernel, activation=func))
    for k in range(droplayers):
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    for _ in range(layers-1):
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
    data["Target"] = encoder(data,"Target")
    hrdat = splittuple(data,0)

    #code for including ppinterval is commented in case we dont need it acc 66%
    ppinterval = splittuple(data,1)
    X_train,X_test,train_target,test_target = evensampler(hrdat)

    #put in case you need ppinterval
    #X_train2,X_test2,_,_ = evensampler(ppinterval)
    #X_train = np.dstack((X_train,X_train2))
    #X_test = np.dstack((X_test,X_test2))
    #X_train=X_train[:,:1300,:]
    #X_test =X_test[:,:1300,:]

    
    #X_train = np.asarray(X_train).astype("float32")
    #X_test = np.asarray(X_test3).astype("float32")
    params = ["convlayers","filter","batches","dropout","func","optimizers"]
    ranges = [[1,2,3,4],[2,4,8,16,32,64],[11,32,64,128],[0.1,0.2,0.3,0.4,0.5],['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']]
    vals = permuter(ranges)
    print(len(vals))

    
    with open('GraduationProject/Emognition/parameter_logs', 'w') as sys.stdout:
        print(tuning(X_train,train_target,X_test,test_target,"cnn1d",(X_train.shape[1],1),vals,params))
#    vals = list(splitter(vals,4))
#
#    with concurrent.futures.ThreadPoolExecutor() as executor:
#        executor.map(,vals)
