import numpy as np
import pandas as pd 
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import multiprocessing as mp
import itertools
def comp(df):

def normcomparer(series1,series2):
   #extracting the actual signals without any padding using the lengths column in the data
   series1 = series1.iloc[2:series1["lengths"]].values.reshape((-1,1))
   series2 = series2.iloc[2:series2["lengths"]].values.reshape((-1,1))
   #lambda x,y: np.linalg.norm(x-y,ord = 1))
   distance ,_= fastdtw(series1,series2,dist = euclidean)
   return distance
def normcomparer2(series1,series2):
   splitted = splitter(series2)
   distancedist = np.zeros(shape = (len(splitted),))
   print(distancedist.shape)
   for i,elem in enumerate(splitted):
      distance ,_= fastdtw(series1,elem,dist = euclidean)
      distancedist[i] = distance  
   #lambda x,y: np.linalg.norm(x-y,ord = 1))
   return distancedist
def splitter(data: pd.Series,window = 50):
   arr_to_split = data.iloc[2:data["lengths"]].values
   splitted = np.array_split(arr_to_split,len(arr_to_split)/window)
   return splitted
def comparer(df,new,total,offset):
      temp = df 
      dists = []
      for i,index in enumerate(new):
         #adding participant and target for the shape of the dataframe to be a readable matrix
         tt = [df.loc[index,"Participant"],df.loc[index,"Target"]]
         for k,j in enumerate(total):
            #basically tells the signal to compare whatever comes after it in the list
            #and resulting in an upper triangular matrix and not repeating the same comparisons
            #more than once
            if k <= i+offset:
               tt.append(0)
            else:
               tt.append(normcomparer(temp.loc[index],temp.loc[j]))
         print(tt)
         dists.append(tt)
      return dists
if __name__ == "__main__":
   hrdat = pd.read_pickle("GraduationProject/Emognition/lengthized.pkl")
   mainmask = hrdat["Stage"] == "STIMULUS"
   uniqemotions = hrdat["Target"].unique()
   finaldf = []
   for emotion in uniqemotions:
      targetmask = hrdat["Target"] == emotion
      dat = hrdat.loc[mainmask&targetmask,:]
      workers = mp.cpu_count()
      print(dat.index)
      #creating a slice of indeces each to run 
      chunks = np.array_split(dat.index,workers)
      pool = mp.Pool(workers)
      #lenghts of each section in the divided indeces 
      lengths = [len(j) for j in chunks]
      #determining offsets to reduce computation and create an upper triangular matrix
      #this uses the prefix sum for the lengths of the indeces to know which indeces
      #that we should compare to which it wouldnt be useful to compare the signal more than once since it's an intensive task
      #this only works knowing that indeces go out ordered when inputted to the function
      offsets = [0,*[sum(lengths[:i+1]) for i in range(len(lengths)-1)]]
      #apply the function on the processors asynchronized since they're not dependent segments
      #no input of a function is dependent on the other
      results = [pool.apply_async(comparer, args = (dat,chunk,dat.index,offset)) for chunk,offset in zip(chunks,offsets)]
      #deleting all working pools
      pool.close()
      #waiting for all to finish and reassigning them
      pool.join()
      #creating a generator object running the function for each segment on processors 
      dist = [result.get() for result in results]
      #finally creating a list from the created generator object where each element of the list
      #is a generator object of the function ran on each segment of the indeces on each segment of the processor 
      dists = list(itertools.chain.from_iterable(dist))
      print(f"Emotion is {emotion}: {dists}")
      finaldf.extend(dists)
   df = pd.DataFrame(data = finaldf,columns = ["Participant","Target",*hrdat["Participant"].unique()])
   df.to_pickle("GraduationProject/Emognition/dtwmat.pkl")