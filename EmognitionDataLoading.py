import json
import os
import pandas as pd
import numpy as np
from operator import itemgetter
path = "E:/ABDO/Graduation project/Datasets/Emognition/_study_data/"
#movie_order will be zipped to an index mapping later after finising the dir
#questionnares will be switched where the movie key will be its value and the value will be the emotion 
#and vad combined where the vad values' key is "sam"
directors = []
directors = os.listdir(path)
print(directors)
experiment = ["STIMULUS","QUESTIONNAIRES"]
maps = ["VALENCE","AROUSAL","MOTIVATION"]
srings = ["heartRate","PPInterval","BVPRaw","BVPProcessed"]
getter = ["AWE", "DISGUST", "SURPRISE", "ANGER", "ENTHUSIASM", "LIKING", "FEAR", "AMUSEMENT", "SADNESS"]
cols = ["Participant","Stage",*srings,"Target",*getter,*maps]
Hzmap = {"heartRate":0.099791,"PPInterval":0.099791,"BVPRaw":0.049898,"BVPProcessed":0.049898}
rows = []
i=0
for dir in directors:
    question = os.path.join(f"{path}/{dir}/{dir}_QUESTIONNAIRES.json")
    vad = {}
    f = open(question)
    data = json.load(f)
    order = data["metadata"]["movie_order"]
    for vads in data["questionnaires"]:
        vad[vads["movie"]] = [vads["emotions"],vads["sam"]]
    ###############10/20/2022
    for key in order:    
        for exp in experiment:
            try:
                rows.append([])
                file = open(f"{path}/{dir}/{dir}_{key}_{exp}_SAMSUNG_WATCH.json")
                dat = json.load(file)
                print(f"{path}/{dir}/{dir}_{key}_{exp}_SAMSUNG_WATCH.json")
                rows[i].append(dir)
                rows[i].append(exp)
                #with resetting timer
                #for string in srings:
                #    dataf= pd.DataFrame(dat[string]).set_index(0).T
                #    col = list(f"{int(x//60)}:{float(x%60)}" for x in np.arange(0,len(dataf.columns)*Hzmap[string],Hzmap[string]))
                #    dataf.rename(columns = dict(zip(dataf.columns,col)),inplace=True)
                #    rows[i].append(dataf)
                #only works on making the data without resetting the timer
                dataframes = list(map(lambda x :pd.DataFrame(x).set_index(0).T,list(itemgetter(*srings)(dat))))
                rows[i].append(key)
                categ = list(itemgetter(*getter)(vad[key][0]))
                cont = list(itemgetter(*maps)(vad[key][1]))
                rows[i].extend(categ)
                rows[i].extend(cont)
                print("complete")
                i += 1
            except:
                continue

final = pd.DataFrame(data = rows,columns = cols)
print(final.loc[0,"heartRate"]['0:0.0'])
final.to_pickle("E:/ABDO/Graduation project/Datasets/Emognition/final(dataframereset).pkl")
