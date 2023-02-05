import concurrent.futures
import inspect
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from Datacheck import encoder, evensampler, sampler, splittuple
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedKFold,
                                     StratifiedKFold)


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
        kfolds = StratifiedKFold(n_splits = cv,random_state = random.randint(0,60000))
        #gets the avg of the kfolds
        avggloballocal = 0
        #tries to get the bestinstance of all kfolds which doesnt really matter but helps you save the mdel
        #although the important thign are the parameters
        bestinstance = 0
        #used to compare accuracy of instances needs to be reinitialized everytime
        instance =0

        print(kfolds.get_n_splits(X))
        for train,test in kfolds.split(X,Y):
            X_train ,X_test = X[train],X[test]
            train_target,test_target = Y[train],Y[test]
            mod = eval(f"{model}({shape},{inputstr})",basedict)

            history,evaluation = fittertester(mod,X_train,train_target
            ,X_test,test_target,batches = basedict["batches"],epoch = basedict["epochs"],metric = "accuracy")
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

#state compiled doesnt really make a difference after testing but left in case it helps further down the road
def cnn1d(shape,state = "compiled",convlayers=1,layers = 2,unit = 100,func="softsign",output=11,filter = 64,kernel = 21,initializer = "he_uniform",
dropout= 0.5,droplayers =1,lossfun = "sparse_categorical_crossentropy",optimizers = "Adamax",metric = ["accuracy"]):
    model = tf.keras.Sequential()
    #kernel_initializer = "glorot_uniform" by default
    model.add(tf.keras.layers.Conv1D(filters = filter,kernel_initializer=initializer,kernel_size=kernel,activation=func,input_shape = shape))
    for i in range(convlayers):
        if i < convlayers-1:
            model.add(tf.keras.layers.Conv1D(filters=filter, kernel_size=kernel, activation=func))
        for k in range(droplayers):
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=5))
    model.add(tf.keras.layers.Flatten())
    if state == "compiled":
        for _ in range(layers):
            model.add(tf.keras.layers.Dense(units = unit,activation=func))
        model.add(tf.keras.layers.Dense(output,activation="softmax"))
        model.compile(loss=lossfun, optimizer=optimizers, metrics= metric)
    return model

def lstmo(shape,state= "compiled",lstmlayer = 4,layers = 1,unitnormal = 100,func="relu",output=11
,unitlstm = 50,dropout= 0.5,lossfun = "sparse_categorical_crossentropy",optimizers = "adam",metric = ["accuracy"]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=unitlstm,return_sequences=True,input_shape=shape))
    model.add(tf.keras.layers.Dropout(dropout))
    for _ in range(lstmlayer-1):
        model.add(tf.keras.layers.LSTM(units = unitlstm,return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Flatten())
    if state == "compiled":
        for _ in range(layers-1):
            model.add(tf.keras.layers.Dense(units = unitnormal,activation=func))
        model.add(tf.keras.layers.Dense(units = output,activation = "softmax"))
        model.compile(loss=lossfun, optimizer=optimizers, metrics= metric)
    return model

#global identifier for metric measures to not be instantiated everytime function called
verbo = {"accuracy":1,"loss":0,"precision":2,"recall":3}
def fittertester(mod,train,train_target,test,test_target,batches = 32,epoch = 50,metric = "accuracy"):
    
    history = mod.fit(train,train_target,validation_split = 0.2
        ,batch_size = batches,epochs = epoch,verbose = 1
        ,callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = 5,
                restore_best_weights=True
            )
        ]
    )
    evaluation = mod.evaluate(test,test_target,verbose = 0)[verbo[metric]]
    return history,evaluation

#tihs works only on 1d cnn not even 2d neds to be optimized but idk how to rn
def layervisualizer(model,layer,sample,path= "./",color = "Spectral"):
    indices = []
    for i,layers in enumerate(model.layers):
        if layer in layers.name:
            #get the index of the desired layer needed to visualize
            indices.append(i)
    #get_weights returns filters,biases so i get 0 because i want the filters weights only
    filters = np.array([model.layers[ind].get_weights()[0] for ind in indices])
    print(filters.shape[3])
    for i in range(len(filters)):
        for j in range(filters.shape[3]):
            #this just means taking all the input values into consideration for each jth filter and ith layer
            f = filters[i,:,:,j]
            fig1 = plt.subplot(8,8,j+1)
            fig1.set_xticks([])
            fig1.set_yticks([])
            plt.imshow(f,cmap = color,aspect = "auto")
        plt.savefig(f"{path}viz_{layer}_layer_{i+1}.png",dpi = 1000)
        plt.close()
    
    output = [model.layers[ind].output for ind in indices]
    vizmodel = tf.keras.Model(inputs = model.inputs,outputs = output)
    outp = vizmodel.predict(sample)
    print(outp.shape)
    #the for i in range is because the shape is (1,6270,64) and if i iterate
    #over the outp itll only iterte for one time and wont iterate over each of the filters
    for j in range(len(outp)):
        for i in range(outp.shape[2]):
            f = outp[:,:,i]
            #subplot starts from 1
            fig1 = plt.subplot(8,8,i+1)
            fig1.set_xticks([])
            fig1.set_yticks([])
            plt.imshow(f,cmap = color,aspect = "auto")
    plt.savefig(f"{path}baseline_thru_{layer}_layer_{j+1}.png",dpi = 1000)
    plt.close()

#abstracted into its own function because it only needs to be done once
#and the multiheaded model could be initialized in the kfolds as its own module
#if it doesnt work then probably each of the head models should be reinitialized everytime the multiheaded model is called
#and if that happens the function built should be modified
def flatter(models):
    index = []
    for model in models:
        for i,layers in enumerate(model.layers):
            if "flatten" in layers.name:
                #get the index of the desired layer needed to visualize
                index.append(i)
    return index
#im sure even if its not a problem it initializes the same random weights for each head
#in each run so i dont know how much of a problem is it but it is only random on the heads
#to remove that we have to reinitialize each model in each kfold as mentioned in the above section
def headjoiner(models: list,index,layers = 1,unit = 100,func = "softsign",output = 11,lossfun = "sparse_categorical_crossentropy",optimizers = "Adamax",metric = ["accuracy"]):
    #so why is it called by outputs makes it work for some reason i still dont know
    #but it has something to do with teh sequential and functional api
    #didnt work in this code
    """
    produced by using this following code

    def headjoiner(models: list,index,layers = 1,unit = 100,func = "softsign",output = 11,lossfun = "sparse_categorical_crossentropy",optimizers = "Adamax",metric = ["accuracy"]):
    flats = [mod.layers[ind].output for ind,mod in zip(index,models)]
    print(models[0].layers[index[0]]._inbound_nodes)
    flats = np.array(flats)
    ###########################################################
    #this istn considered as aviable input needs to be checked
    merged = tf.keras.layers.concatenate(inputs = flats,axis =-1)
    #couldnt use it as add since merged isnt considered as a model you could add to 
    for _ in range(layers):
        merged = tf.keras.layers.Dense(units = unit,activation=func)(merged)
    merged = tf.keras.layers.Dense(output,activation="softmax")(merged)
    inputs = [model.inputs for model in models]
    mergedmod = tf.keras.Model(input = inputs,output = merged)
    mergedmod.compile(loss=lossfun, optimizer=optimizers, metrics= metric)
    return mergedmod

    TypeError: You are passing KerasTensor(type_spec=TensorSpec(shape=(None, 80256), 
    dtype=tf.float32, name=None), name='flatten_16/Reshape:0', description="created by layer 'flatten_16'"),
    an intermediate Keras symbolic input/output, to a TF API that does not allow registering custom dispatchers,
    such as `tf.cond`, `tf.function`, gradient tapes, or `tf.map_fn`. Keras Functional model construction
    only supports TF API calls that *do* support dispatching, such as `tf.math.add` or 
    `tf.reshape`. Other APIs cannot be called directly on symbolic Kerasinputs/outputs. 
    You can work around this limitation by putting the operation in a custom 
    Keras layer `call` and calling that layer on this symbolic input/output
    
    #might be helpful
    https://stackoverflow.com/questions/44042173/concatenate-merge-layer-keras-with-tensorflow
    
    https://keras.io/guides/functional_api/

    https://www.google.com/search?q=How+to+%22Merge%22+Sequential+models+in+tensorflow&ei=VkrfY97XGqvUkdUP0IiOkAY&ved=0ahUKEwje2rWg3v38AhUraqQEHVCEA2IQ4dUDCA8&uact=5&oq=How+to+%22Merge%22+Sequential+models+in+tensorflow&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCCEQoAE6CAgAEIYDELADSgQIQRgASgQIRhgAUPUEWLMGYJwIaABwAHgAgAGXAYgBhwOSAQMwLjOYAQCgAQHIAQXAAQE&sclient=gws-wiz-serp
    
    really useful link as well:
    https://stackoverflow.com/questions/53942291/what-does-the-00-of-the-layers-connected-to-in-keras-model-summary-mean

    explains how each layer is a node and carries the structure along wiht it no need to 
    """
    flats = [mod.layers[ind].output for ind,mod in zip(index,models)]
    print(models[0].layers[index[0]]._inbound_nodes)
    ###########################################################
    #this istn considered as aviable input needs to be checked
    merged = tf.keras.layers.Concatenate()(flats)
    #couldnt use it as add since merged isnt considered as a model you could add to 
    for _ in range(layers):
        merged = tf.keras.layers.Dense(units = unit,activation=func)(merged)
        merged = tf.keras.layers.Dropout(0.5)(merged)
    merged = tf.keras.layers.Dense(output,activation="softmax")(merged)
    inputs = [model.inputs for model in models]
    
    mergedmod = tf.keras.Model(inputs,merged)
    mergedmod.compile(loss=lossfun, optimizer=optimizers, metrics= metric)
    return mergedmod

def resultviz(path,model,hist,actual,test,uniques):
    #predictions is the max probability over the predictions since each prediction element is an array of length 11 with probvabliities of each class
    prediction = np.array(list(map(lambda x:np.argmax(x),model.predict(test))))
    cs = confusion_matrix(actual,prediction)
    clr = classification_report(actual,prediction,target_names = uniques,output_dict=True)
    plt.figure(figsize = (10,10))
    sns.heatmap(cs,annot = True,cmap = "GnBu")
    plt.xticks(np.arange(len(uniques)),remapper)
    plt.yticks(np.arange(len(uniques)),remapper)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{path}Heatmap.png")
    print(model.summary())
    plt.close()
    plt.plot(hist.history["loss"],label="train loss")
    plt.plot(hist.history["val_loss"],label="validation loss")
    plt.legend()
    plt.savefig(f"{path}Trainingloss.png")
    plt.close()
    plt.plot(hist.history['accuracy'],label='accuracy')
    plt.savefig(f"{path}accuracy.png")
    print("here")
    dataframe = pd.DataFrame(clr)
    dataframe.to_csv(f"{path}{model.layers[0].name}_report.csv")

if __name__ == "__main__":
    data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
    #it is turned into an object i dont know why but it's turned into an inta nd type changed before
    #mask so i dont pass it through another func
    data = data[data["Stage"] == "STIMULUS"]
    remapper = data["Target"].unique()
    data["Target"] = encoder(data,"Target")
    data = splittuple(data,0)
    databvp = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/bvp(tuples).pkl")
    print(databvp)
    databvp = databvp[databvp["Stage"] == "STIMULUS"]
    databvp = splittuple(databvp,1)
    # get the target values
    #put in an np.array as int idk why but it used to work without it now it 
    #reads its type not as an int but as an object while every var inside is an instance of an int

#data splitting, and model prepping
########################################################################################################################################################
    datatarget = np.asarray(data["Target"].values).astype(int)
    #adding pp interval is dstacking two splits one split for the hr and one for the ppinterval
    databvp = np.asarray(databvp.values[:,2:-13]).astype(float)
    data =  np.asarray(data.values[:,2:-13]).astype(int)

    model1 = cnn1d((data.shape[1],1))
    model2 = cnn1d((databvp.shape[1],1))
    #just for ease in writing
    modlist = [model1,model2]
    index = flatter(modlist)
    
#model training, and visualization
###########################################################################################################################################################   
    kfolds = StratifiedKFold(n_splits=2)
    models = []
    accuracy = []
    histories = []
    tests = [] 
    targets = []
    for train,test in kfolds.split(data,datatarget):
            X_train,X_test = [data[train],databvp[train]],[data[test],databvp[test]]
            train_target,test_target = datatarget[train],datatarget[test]
            mod = headjoiner(models = modlist,index = index,layers = 3)
            history,evaluation = fittertester(mod,X_train,train_target,X_test,test_target)
            accuracy.append(evaluation)
            histories.append(history)
            tests.append(X_test)
            models.append(mod)
            targets.append(test_target)
    print(f"Model with best Parameters on stimuli:{np.mean(accuracy)},Standard Deviation:{np.std(accuracy)}")
    ind = np.argmax(accuracy)
    testhist = histories[ind]
    test = tests[ind]
    model = models[ind]
    target = targets[ind]
    resultviz(path = "E:/ABDO/Courses/Python/GraduationProject/Emognition/",model = model,hist = testhist,actual = target,test = test,uniques = remapper)
########################################################################################################################################################


#convulotion layers filters and outputs for patterns visualized
########################################################################################################################################################
    tempdat = np.expand_dims(data[0],axis = 0)
    layervisualizer(models[ind],"conv",sample = tempdat,path = "E:/ABDO/Courses/Python/GraduationProject/Emognition/")
    
########################################################################################################################################################

#parameter tuning which requires 
########################################################################################################################################################
    #code for including ppinterval is commented in case we dont need it acc 66%
    #ppinterval = splittuple(data,1)
    #commenting to try cv
    #X_train,X_test,train_target,test_target = evensampler(hrdat)

    #put in case you need ppinterval
    #X_train2,X_test2,_,_ = evensampler(ppinterval)
    #X_train = np.dstack((X_train,X_train2))
    #X_test = np.dstack((X_test,X_test2))
    #X_train=X_train[:,:1300,:]
    #X_test =X_test[:,:1300,:]

#tuning plus a trial of parallel processing which wasnt successful
########################################################################################################################################################
    #params = ["convlayers","filter","func","optimizers"]
    #ranges = [[1,2,3,4],[16,32,64],['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid'],[ 'Adagrad', 'Adam', 'Adamax', 'Nadam']]
    #vals = permuter(ranges)
    #print(len(vals))
    #result = tuning(hrdat,hrtarget,"cnn1d",(hrdat.shape[1],1),vals,params)

    #with open('GraduationProject/Emognition/parameter_logs(cv).txt', 'w') as sys.stdout:
    #    print(list(result))


    #print(vals)
    #print(len(vals))
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #   result = executor.map(lambda x:tuning(hrdat,hrtarget,"cnn1d",(hrdat.shape[1],1),x,params),vals)
########################################################################################################################################################
