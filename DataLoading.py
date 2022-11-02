import json
import os
import pandas as pd
import numpy as np
from operator import itemgetter

#path to the directory of the eomgnition files
path = "E:/ABDO/Graduation project/Datasets/Emognition/_study_data/"
#participants in the directory
directors = os.listdir(path)
print(directors)
#stages of the experiment usually washout then stimulus then questionnares
experiment = ["WASHOUT","STIMULUS","QUESTIONNAIRES"]
#maps will be used to extract the vam at once from the questionnare file
maps = ["VALENCE","AROUSAL","MOTIVATION"]
#Needed signals from the file
srings = ["heartRate","PPInterval","BVPRaw","BVPProcessed"]
#used to extract the discrete emotional values from the questionnare
getter = ["AWE", "DISGUST", "SURPRISE", "ANGER", "ENTHUSIASM", "LIKING", "FEAR", "AMUSEMENT", "SADNESS"]
#frequencies of the heartrate,ppinterval and the bvp(raw,processed) 0 hr 1 bvp
hzmap = [0.099791,0.049898]
#the frames of the hr and the bvp divided into two
hr = []
bvp = []
emotions = []
lenhr = 0
lenbvp = 0
i=0
for dir in directors:
    #getting the questionnare file for every participant
    question = os.path.join(f"{path}/{dir}/{dir}_QUESTIONNAIRES.json")
    #a dictionary for mapping the vad,categorical for each movie due to the difficulty for accessing the questionnare file mapping
    vad = {}
    f = open(question)
    #json.load turns the json file into a huge dictionary of dictionaries
    data = json.load(f)
    #getting the movie order so its order in the same order in the df
    order = data["metadata"]["movie_order"]
    #making the mapping instead of "movie":"movie type","emotion":discrete,"sam":vam into "movie type":[{discrete},{V: ,A: ,M:}] 
    for vads in data["questionnaires"]:
        vad[vads["movie"]] = [vads["emotions"],vads["sam"]]
    ###############10/20/2022
    for key in order:    
        for exp in experiment:
            try:
                file = open(f"{path}/{dir}/{dir}_{key}_{exp}_SAMSUNG_WATCH.json")
                dat = json.load(file)
                #the append after file loading bec if any error happens it wouldnt add an empty row
                hr.append([])
                bvp.append([])
                emotions.append([])
                print(f"{path}/{dir}/{dir}_{key}_{exp}_SAMSUNG_WATCH.json")
                #append the first col which is participant
                hr[i].append(int(dir))
                bvp[i].append(int(dir))
                #append the experiment stage
                hr[i].append(exp)
                bvp[i].append(exp)
                #the itemgetter srings(dat) returns a 3d list where the first dimension is the signal second is the timestamp or value(cols) third is the rows
                #list(map(itemgetter(1)),x) returns the second column of each element in each signal
                frames = list(map(lambda x : list(map(itemgetter(1),x)),list(itemgetter(*srings)(dat))))
                #zipping the hr and ppinterval into its own list of tuples
                hrpp = list(map(tuple,zip(frames[0],frames[1])))
                #zipping raw and processed bvp into its own list of tuples
                procbvp = list(map(tuple,zip(frames[2],frames[3])))
                #finding the max watched length video and its corresponding values
                lenhr = max(lenhr,len(hrpp))
                lenbvp = max(lenbvp,len(procbvp))
                #extend because we're adding the elements of the hrpp or procbvp to the row not the actual list
                hr[i].extend(hrpp)
                bvp[i].extend(procbvp)
                #appending the keys in emotions on its own so it doesnt interfere with the extra timestamps
                emotions[i].append(key)
                #getting all the categorical and vam values from the frame initialized above
                categ = list(map(int,list(itemgetter(*getter)(vad[key][0]))))
                cont = list(map(int,list(itemgetter(*maps)(vad[key][1]))))
                emotions[i].extend(categ)
                emotions[i].extend(cont)
                print("complete")
                i += 1
            except:
                continue

#timestamps with mins and secs from 0 till the max time either hr or bvp
hrfreq = list(f"{int(x//60)}:{float(x%60)}" for x in np.arange(0,lenhr*hzmap[0],hzmap[0]))
bvpfreq = list(f"{int(x//60)}:{float(x%60)}" for x in np.arange(0,lenbvp*hzmap[1],hzmap[1]))
print(hrfreq)
print(bvpfreq)
emotioncols = ["Target",*getter,*maps]
hrcols = ["Participant","Stage",*hrfreq]
bvpcols = ["Participant","Stage",*bvpfreq]
hrdat = pd.DataFrame(data = hr,columns = hrcols)
print(hrdat)
bvpdat = pd.DataFrame(data = bvp,columns = bvpcols[:-1])

hrdat[emotioncols] = emotions
bvpdat[emotioncols] = emotions

hrdat.to_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
bvpdat.to_pickle("E:/ABDO/Graduation project/Datasets/Emognition/bvp(tuples).pkl") 
