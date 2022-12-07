import tensorflow as tf
import pandas as pd
import numpy as np
from Datacheck import sampler,splittuple,encoder
from sklearn.preprocessing import normalize

def cnn1d(shape,convlayers=2,layers = 2,unit = 100,func="relu",output="11",filter = 64,kernel = 3,dropout= 0.5):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters = filter,kernel_size=kernel,activation=func,input_shape = shape))
    for _ in range(convlayers-1):
        model.add(tf.keras.layers.Conv1D(filters=filter, kernel_size=kernel, activation=func))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    for _ in range(layers-1):
        model.add(tf.keras.layers.Dense(units = unit,activation=func))
    model.add(tf.keras.layers.Dense(output,activation="softmax"))
    return model
def lstmo(shape,lstmlayer = 4,layers = 1,unitnormal = 100,func="relu",output="11",unitlstm = 50,dropout= 0.5):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=unitlstm,return_sequences=True,input_shape=shape))
    model.add(tf.keras.layers.Dropout(0.2))
    for _ in range(lstmlayer-1):
        model.add(tf.keras.layers.LSTM(units = unitlstm,return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    for _ in range(layers-1):
        model.add(tf.keras.layers.Dense(units = unitnormal,activation=func))
    model.add(tf.keras.layers.Dense(units = 11,activation = "softmax"))
    return model
if __name__ == "__main__":
    data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
    data["Target"] = encoder(data,"Target")
    hrdat = splittuple(data,0)
    print(data)
    #code for including ppinterval is commented in case we dont need it acc 66%
    ppinterval = splittuple(data,1)
    X_train,X_test,train_target,test_target = sampler(hrdat)
    #put in case you need ppinterval
    #X_train2,X_test2,_,_ = sampler(ppinterval)
    #X_train = np.dstack((X_train,X_train2))
    #X_test = np.dstack((X_test,X_test2))
    
    #X_train = np.asarray(X_train).astype("float32")
    #X_test = np.asarray(X_test3).astype("float32")
    print(X_train.shape)
    print(X_test.shape)
    print(train_target.shape)
    print(test_target.shape)
    #replace with 1dcnn if you want to try
    model = lstmo((X_train.shape[1],1),lstmlayer = 2)
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train,train_target,validation_split = 0.2
        ,batch_size = 11,epochs = 50
        ,callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = 5,
                restore_best_weights=True
            )
        ]
    )
    print(model.evaluate(X_test,test_target,verbose=0)[1])





