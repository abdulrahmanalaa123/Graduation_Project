import pandas as pd
from Datacheck import splittuple
import numpy as np
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import sys
def correlator(x:pd.Series,index,data:pd.DataFrame):
    cutoff = x["lengths"]
    base = x[data.columns[2:cutoff]].values
    washout = data.loc[index-1,data.columns[2:2+cutoff]].values
    quest = data.loc[index+1,data.columns[2:2+cutoff]].values
    w = np.correlate(base,washout,"full")
    q = np.correlate(base,quest,"full")

    return w,q

def increase(dict1,counter):
    dict1 = {key:dict1.get(key,0) + counter[key] for key in counter.keys()}
    return dict1
data = pd.read_pickle("hr(tuples).pkl")

hrdat = splittuple(data,0)
print(hrdat)
print(hrdat["Stage"].unique())
hrdat["lengths"] = np.sum(hrdat[hrdat.columns[2:-13]].apply(lambda x:x > 0).values,axis = 1)

print(hrdat.index)
print(hrdat["Target"].unique())
#put the mask for baseline although i think it's bad but will make the iterable eaiser
iterav = hrdat[(hrdat["Stage"] == "STIMULUS") & (hrdat["Target"] != "BASELINE")]
ind = iterav.index

#alll commented codes was to get the freq and now aree uselless
washques = []
shiftwashques = []
for i in ind:
    corr = correlator(iterav.loc[i],i,hrdat)
    maxwash = np.max(corr[0])
    maxques = np.max(corr[1])
    if  maxwash> 6500000:
        shiftwash = np.argmax(corr[0])
        maxwash =True
    else:
        maxwash =False
        shiftwash = 0
    if maxques > 2000000:
        shiftques = np.argmax(corr[0])
        maxques = True
    else:
        maxques = False
        shiftques = 0
    washques.append([maxwash,maxques])
    shiftwashques.append([shiftwash,shiftques])
    print(maxwash,maxques)
   
    
for l,i in enumerate(ind):
    base = hrdat.loc[i-1,"lengths"]
    trans = abs(shiftwashques[l][0]-base)
    if washques[l][0] == 1 and shiftwashques[l][0]<6290:
        tmp = hrdat.loc[i-1].values
        allsign = tmp[2:2+base]
        tmp[2+trans:2+base+trans] = allsign
        tmp[2:2+trans] = [0]*trans
        hrdat.loc[i-1] = tmp
    else:
        tmp2 = hrdat.loc[i-1].values
        tmp2= [0]*(len(tmp2))
        hrdat.loc[i-1] = tmp2
    
    base = hrdat.loc[i+1,"lengths"]
    trans = abs(shiftwashques[l][1]-base)
    if washques[l][1] == 1 and shiftwashques[l][1]<6290:
        tmp = hrdat.loc[i+1].values
        allsign = tmp[2:2+base]
        tmp[2+trans:2+base+trans] = allsign
        tmp[2:2+trans] = [0]*trans
        hrdat.loc[i+1] = tmp
    else:
        tmp2 = hrdat.loc[i+1].values
        tmp2= [0]*(len(tmp2))
        hrdat.loc[i-1] = tmp2
hrdat.to_pickle("C:/Users/AbdelRahman Rashed/Desktop/work/shifteddata2mil_6.5mil.pkl")
#fig = plt.figure(figsize=(10,10))
#print("here")
#plt.boxplot([wash,qus])

#plt.savefig("GraduationProject/Emognition/boxqus.png")
#print(qfreq)
#qfreq.to_pickle("GraduationProject/Emognition/questionarecrossfreq.pkl")
