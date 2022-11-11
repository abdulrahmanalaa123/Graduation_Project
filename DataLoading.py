import json
import os
import pandas as pd
import numpy as np
from operator import itemgetter
from collections import defaultdict
import re
path = "E:/ABDO/Graduation project/Datasets/Emognition/_study_data/"
#movie_order will be zipped to an index mapping later after finising the dir
#questionnares will be switched where the movie key will be its value and the value will be the emotion 
#and vad combined where the vad values' key is "sam"
directors = []
directors = os.listdir(path)
print(directors)
experiment = ["WASHOUT","STIMULUS","QUESTIONNAIRES"]
maps = ["VALENCE","AROUSAL","MOTIVATION"]
srings = ["heartRate","PPInterval","BVPRaw","BVPProcessed"]
getter = ["AWE", "DISGUST", "SURPRISE", "ANGER", "ENTHUSIASM", "LIKING", "FEAR", "AMUSEMENT", "SADNESS"]
hzmap = [0.099791,0.049898]
hr = []
bvp = []
emotions = []
times = {}
time_freq = defaultdict(int)
lenhr = 0
lenbvp = 0
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
                file = open(f"{path}/{dir}/{dir}_{key}_{exp}_SAMSUNG_WATCH.json")
                dat = json.load(file)
                hr.append([])
                bvp.append([])
                emotions.append([])
                print(f"{path}/{dir}/{dir}_{key}_{exp}_SAMSUNG_WATCH.json")
                hr[i].append(int(dir))
                bvp[i].append(int(dir))
                hr[i].append(exp)
                bvp[i].append(exp)
                lengths  = list(map(lambda x: len(x),list(itemgetter(*srings)(dat))))   
                if exp == "STIMULUS":
                    time_freq[lengths[0]] += 1
                    if times.get(key) is None:
                        times[key] = []
                    times[key].append(f"participant:{dir}:{lengths[0]}") if lengths[0] not in list(map(lambda x:int(re.findall(r'\d+',x[-4:])[0]),times[key])) else print("repeated")
                frames = list(map(lambda x : list(map(itemgetter(1),x)),list(itemgetter(*srings)(dat))))
                hrpp = list(map(tuple,zip(frames[0],frames[1])))
                procbvp = list(map(tuple,zip(frames[2],frames[3])))
                lenhr = max(lenhr,len(hrpp))
                lenbvp = max(lenbvp,len(procbvp))
                print(lenhr,lenbvp)
                hr[i].extend(hrpp)
                bvp[i].extend(procbvp)
                emotions[i].append(key)

                categ = list(map(int,list(itemgetter(*getter)(vad[key][0]))))
                cont = list(map(int,list(itemgetter(*maps)(vad[key][1]))))
                emotions[i].extend(categ)
                emotions[i].extend(cont)
                print("complete")
                i += 1
            except:
                continue

hrfreq = list(f"{int(x//60)}:{float(x%60)}" for x in np.arange(0,lenhr*hzmap[0],hzmap[0]))
bvpfreq = list(f"{int(x//60)}:{float(x%60)}" for x in np.arange(0,lenbvp*hzmap[1],hzmap[1]))

emotioncols = ["Target",*getter,*maps]
hrcols = ["Participant","Stage",*hrfreq]
bvpcols = ["Participant","Stage",*bvpfreq]
hrdat = pd.DataFrame(data = hr,columns = hrcols)

bvpdat = pd.DataFrame(data = bvp,columns = bvpcols[:-1])

hrdat = hrdat.fillna(0)
bvpdat = bvpdat.fillna(0)
hrdat[emotioncols] = emotions
bvpdat[emotioncols] = emotions
print(times)
print(time_freq)
hrdat.to_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
bvpdat.to_pickle("E:/ABDO/Graduation project/Datasets/Emognition/bvp(tuples).pkl") 
