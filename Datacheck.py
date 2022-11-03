from re import L
import pandas as pd
import json
import numpy as np
from operator import itemgetter
def getter(x):
    if isinstance(x,tuple):
        return itemgetter(0)(x)
    else:
        return x
        
data = pd.read_pickle("E:/ABDO/Graduation project/Datasets/Emognition/hr(tuples).pkl")
print(data.dtypes)
mask = data["Participant"] ==  22 
mask2 = data["Stage"] == "STIMULUS"
print(data.columns[-13:])
print(data.loc[mask & mask2,[*data.columns[2:500],*data.columns[-13:]]].applymap(lambda x:getter(x)))
