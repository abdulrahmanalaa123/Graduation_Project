from time import time
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import sklearn.preprocessing as sk



path = "E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Raw Data/Single Modal/ECG"
matfiles = glob.glob(f"{path}/*.mat")
array = np.arange(0,5000)
df = pd.DataFrame(array.reshape(1,-1))

for paths in matfiles:
    mat = scipy.io.loadmat(paths)
    temp = pd.DataFrame(mat["ECGdata"])
    temp["Session ID"] = int(re.search("(s)\w{1}",paths[-11:])[0][1])
    temp["Video ID"] = int(re.search("(v)\w{1}",paths[-11:])[0][1])
    temp["Participant ID"] = int("".join(re.findall("\d",re.search("(p)\w{2}",paths[-11:])[0])))
    df = df.append(temp)

df = df.iloc[1:,:]


self_data = pd.read_csv("E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Self-Annotation Labels/Single_modal(encoded).csv")
#self_data = self_data.astype({"Arousal level": int})
cols = ['Happy','Sad', 'Fear', 'Anger', 'Neutral', 'Disgust', 'Surprised']
self_data = self_data.rename(columns = {"Participant Id":"Participant ID","Session Id":"Session ID","Video Id":"Video ID"})
print(self_data.columns)
identifiers = ["Participant ID","Session ID","Video ID"]
tuples = list(map(tuple,self_data[identifiers].values))
values = self_data[cols].values
mapper = dict(zip(tuples,values))
print(mapper)

df[cols] = pd.DataFrame(list(map(lambda x:mapper[x],list(map(tuple,(temp[identifiers].values))))))

timestamps = np.arange(0,39,1/256)
renamedict = dict(zip(array,timestamps))
df = df.rename(columns = renamedict)

df.to_csv(f"{path}/concatenated.csv")



    
