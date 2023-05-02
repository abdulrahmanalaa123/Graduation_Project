import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten ,LSTM,RepeatVector,TimeDistributed,Bidirectional
#from keras.layers import  Conv2D, MaxPooling2D
from keras.utils import to_categorical
from re import L
#from keras.utils import get_sequences
from sklearn.preprocessing import StandardScaler

def getter(x,index):
    if isinstance(x,tuple):
        return x[index]
    else:
        return x
def splittuple(dat,index):
    return dat.applymap(lambda x:getter(x,index))

def to_seq(x,y,seq_size=1):
  x_values=[]
  y_values=[]
  for i in range(len(x)-seq_size):
    x_values.append(x.iloc[i:(i+seq_size)].values)
    y_values.append(y.iloc[seq_size])
  return np.array(x_values),np.array(y_values)
def ranges(data):
    if len(data) > 0: 
        diffed = data[1:]-data[:-1]
        indeces = np.where(diffed > 1)[0]
        indeces = np.concatenate([[-1],indeces,[len(data)-1]])
        #after debugging the index is found to be reduced by one so the beginning is added by 1 to the next actual index 
        #since its decreased by one so it's actually the end of the first sequence
        ranges = [f"{data[indeces[i]+1]}:{data[indeces[i+1]]}"for i in range(len(indeces)-1)]
        array = []
        for i in range(len(indeces)-1):
            array.append(data[indeces[i]+1])
            array.append(data[indeces[i+1]]) 
        return ranges,array
    else:
        return [0],[0]
def scaleralligner(train):
      train = train.T
      train = train.rename(columns = {train.columns[0]:0})
      scaler=StandardScaler()
      scaler=scaler.fit(train[[0]])
      train[0]=scaler.transform(train[[0]])
      train['time'] = ['00:'+x   for x in train.index]
      train['time'] = pd.to_timedelta(train['time'])
      return train
def parter(data,rang,seq_size,participant):
    mask3 = data["Target"] == "BASELINE"
    emotions = data["Target"].unique()
    sub =  data.loc[mask3,data.columns[2:rang]]
    train=sub.copy()
    train = scaleralligner(train)
    trainx,trainy=to_seq(train[[0]],train[0],seq_size)

    model=Sequential()
    model.add(LSTM(50,input_shape=(trainx.shape[1],trainx.shape[2]),return_sequences=False))
    #model.add(LSTM(64,activation='relu',return_sequences=False))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(trainx.shape[1]))
    #model.add(LSTM(64,activation='relu',return_sequences=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(trainx.shape[2])))
    model.compile(optimizer='adam', loss='mae',metrics=['accuracy'])
    h=model.fit(trainx,trainy,epochs=20,batch_size=50,validation_split=0.2,verbose=1)
    trainPredict = model.predict(trainx)
    trainMAE = np.mean(np.abs(trainPredict - trainx), axis=1)
    max_trainMAE = trainMAE[-1]
    trainPredict = model.predict(trainx)
    trainMAE = np.mean(np.abs(trainPredict - trainx), axis=1)
    max_trainMAE = trainMAE[-1]
    values = []
    targets = []
    for emotion in emotions[1:]:
        sub2 = data[data["Target"] == emotion]
        sub2 = sub2[sub2.columns[2:rang]]
        final = scaleralligner(sub2)
        #finalx the inputted sliced array
        finalx,finaly =to_seq(final[[0]],final[0],seq_size)
        #array of predictions
        finalpredict = model.predict(finalx,verbose= 0)
        #array
        trainMAE = np.mean(np.abs(finalpredict - finalx), axis=1)
        
        #array of true and falses
        maskforrange = trainMAE > max_trainMAE
        indeces = np.where(maskforrange)[0]

        rangat,array = ranges(indeces)
        for l in range(0,len(array)-1,2):
            temp = []
            temp.append(participant)
            #print("array of ranges : " + str(array))
            #print("target: " + str(emotion))
            #print(f"current range {array[l]}:{array[l+1]}")
            temp.extend(sub2.iloc[[0],array[l]:array[l+1]].values[0])
            values.append(temp)
            targets.append(emotion)
    return values,targets
def anomal(data,seq_size,length):
    participants = data["Participant"].unique()
    mask = data["Stage"] == "STIMULUS"
    final = []
    targets = []
    for participant in participants:
        mask2 = data["Participant"] == participant
        datat,target = parter(data[mask2&mask],length,seq_size,participant)
        final.extend(datat)
        targets.extend(target)
    print(len(final),len(final[0]))
    colums = ["Participant"]
    maxlen = max([len(elem) for elem in final])
    colums.extend([i for i in range(maxlen-1)])
    datafinale = pd.DataFrame(final,columns = colums)
    datafinale["Target"] = targets 
    return(datafinale)
if __name__ == "__main__":
    data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
    hrdat = splittuple(data,0)
    final = anomal(hrdat,30,800)
    #final.to_pickle(.....)
