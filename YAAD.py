import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import sklearn.preprocessing as sk
from scipy.spatial.distance import cosine



path = "E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Raw Data/Single Modal/ECG"
matfiles = glob.glob(f"{path}/*.mat")
array = np.arange(0,5000)
df = pd.DataFrame(array.reshape(1,-1))

def changer(x):
    try:
        x = int(x)
        return x
    except:
        return 8


for paths in matfiles:
    mat = scipy.io.loadmat(paths)
    temp = pd.DataFrame(mat["ECGdata"])
    temp["Session ID"] = re.search("(s)\w{1}",paths[-11:])[0][1]
    temp["Video ID"] = re.search("(v)\w{1}",paths[-11:])[0][1]
    df = df.append(temp)

df = df.iloc[1:,:]
df.to_csv(f"{path}/concatenated.csv")
mapping = pd.read_excel("E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Stimulus_Description.xlsx")


tuples = list(map(tuple,mapping[["Session ID","Video ID"]].values))
emotions = mapping["Target Emotion"].values
mapper = dict(zip(tuples,emotions))
#https://stackoverflow.com/questions/30459485/replacing-row-values-in-pandas
#this link may be useful in the sense of manipulating dataframes and dictionaries and manipulating values
#old code for creating a mapper and zipping was a more efficient way
#print(mapper)
#for _, row in mapping.iterrows():
#    tup = tuple(row[["Session ID","Video ID"]].values)
#    value = row[["Target Emotion"]].values[0]
#    mapper[tup] = value

#turning the session id and video id into int can be done from the origin but ill leave it as it is
#because its something new that I've learned
df = df.astype({"Session ID" : int,"Video ID":int})
df_tuples = list(map(tuple,df[["Session ID","Video ID"]].values))
Target_emotion = list(map(lambda x: mapper[x],df_tuples))

#Changed this method as the mapping is a more efficient way i think than iterating every row

##Problem in iterrows and the loc function must be speculated further
#for _,row in df.iterrows():
#    tup = tuple(list(map(int,row[["Session ID","Video ID"]].values))) 
#    Target_emotion.append(mapper[tup])

df["Target Emotion"] = Target_emotion
#cool idea of making several csvs
DataFrameDict = {elem : pd.DataFrame() for elem in mapping["Target Emotion"].unique()}
#i believe that this piece of code is at its best form
for key in DataFrameDict.keys():
   DataFrameDict[key] =  df.loc[df["Target Emotion"] == key]
   DataFrameDict[key].to_csv(f"{path}/{key}_table.csv")


self_data = pd.read_excel("E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Single Modal.xlsx")
#self_data = self_data.astype({"Arousal level": int})
 

cols = ['Happy','Sad','Fear','Anger','Neutral','Disgust','Surprised']

self_data[cols] = self_data[cols].replace(["VeryLow","Low","Moderate","High","VeryHigh"],[-2,-1,0,1,2])

vadums = ['Valence level', 'Arousal level', 'Dominance level']

self_data["Arousal level"] = self_data["Arousal level"].apply(lambda x:changer(x))
standard_scaler = sk.StandardScaler()
self_data.loc[:,vadums] = standard_scaler.fit_transform(self_data.loc[:,vadums])
print(self_data[vadums])
vad = []
mask = self_data[cols] >= 1
for i,col in enumerate(cols):
    temp = self_data[vadums].loc[mask[col]].mean()
    vad.append(list(temp.values)) 
print(vad)

#for col in cols:
#    plt.figure(figsize = (10,10))
#    ax = plt.axes(projection ="3d")
#    my_cmap = plt.get_cmap('hsv')
#    sctt = ax.scatter3D(self_data.loc[mask[col],"Arousal level"],self_data.loc[mask[col],"Valence level"],self_data.loc[mask[col],"Dominance level"],alpha = 0.8,
#                        cmap = my_cmap,
#                        marker ='^')
#    ax.set_title(col)
#    ax.set_xlabel("Arousal")
#    ax.set_ylabel("Valence")
#    ax.set_zlabel("Dominance")
#    plt.savefig(f"E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Self-Annotation Labels/{col}.png")
#    plt.close()
#
#
#fig,ax = plt.subplots(7,3,figsize=(30,30))
#
#i=0
#for col in cols:
#    ax[i][0].set_title(col)
#    ax[i][0].scatter(self_data.loc[mask[col],"Arousal level"],self_data.loc[mask[col],"Valence level"])
#    ax[i][0].set_xlabel(f"Arousal_{col}")
#    ax[i][0].set_ylabel(f"Valence_{col}")
#    ax[i][1].scatter(self_data.loc[mask[col],"Arousal level"],self_data.loc[mask[col],"Dominance level"])
#    ax[i][1].set_xlabel(f"Arousal_{col}")
#    ax[i][1].set_ylabel(f"Dominance_{col}")
#    ax[i][2].scatter(self_data.loc[mask[col],"Valence level"],self_data.loc[mask[col],"Dominance level"])
#    ax[i][2].set_xlabel(f"Valency_{col}")
#    ax[i][2].set_ylabel(f"Dominance_{col}")
#    i+=1
#
#fig.savefig("E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Self-Annotation Labels/all.png")
#self_data.loc[:,cols[:]].values =
vad = np.array(vad)
norms = np.linalg.norm(vad,axis =1).reshape(7,1)
norms2 = np.linalg.norm(self_data[vadums].values,axis = 1)
cos = np.dot(vad,self_data[vadums].T.values)/norms*norms2

self_data[cols[:]] = np.transpose(cos)[:]

self_data.to_csv("E:/ABDO/Graduation project/Datasets/Kaggle/g2p7vwxyn2-1/ECG_GSR_Emotions/Self-Annotation Labels/Self_annotation(cosinecheck).csv")