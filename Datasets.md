# DataSets

# CASE
## Keywords
Functional Data Analysis
## Notes:
  DAQ: data aquisition file contains all the sensors
  joystick values are the position of the joystick is mapped to [0.5,9.5] for x is valence and y is arousal to map those values
  scripts has been added in the code so we could replicate the interpolation done by the releasers of the dataset which is done because the sampling rate and frequency     of reading wasnt stable in all users so to unify the data interpolation is needed which if learned how to do would help us in the deployment phase
  frequency of the sampling sensors are set to 1k HZ and the joystick is at 20HZ the mapping method is still yet to be figured out
  
The data is a continous annoation of users emotions using VA model while watching a video stimuli(movie) and is done in different orders except the first 2 starting movies which I think is the neutral videos to start off on the same basis and then record the fluctuations and the analysis process is documented in the paper here:
https://www.researchgate.net/publication/331296934_A_functional_data_analysis_approach_for_continuous_2-D_emotion_annotations

## Functional Analysis:
https://anson.ucdavis.edu/~mueller/Review151106.pdf

## useful because
interpolation method,realtime annotation(huge point),analysis method as well if understood,mapping if done(very probable its mentioned in the script sec)
## issues:
valence arousal mapping to discrete emotions (kinda referenced in the paper but not fully understood)

## anything else needed and the dataset itself can be found here:
https://gitlab.com/karan-shr/case_dataset/-/tree/master/

# Emognition
## Brief
  Subjects were issued to watch a movie clip with a baseline video of dots and lines to measure the baseline emotion where between each video and the other each subject is exposed to a 2 min video of a baseline refresher to not mix the emotions between the different stimulis
## Keywords:
Extracting,ECG,BVP from ppg signals,transforming bvp to ecg and its accuracy
## Note:
  the bvp read from the samsung watch was using its ppg sensor which he used the raw input values he got from making the app in the tizen os and applied certain methods mentioned in the methods section in the paper to extract the bvp from them which the same method was used in empatica and samsung where in the empatica the bvp  is calculated using the empatica sdk or (os) while the method of transformation could be found here i think:
  https://support.empatica.com/hc/en-us/articles/204954639-Utilizing-the-PPG-BVP-signal
  https://www.nature.com/articles/s41597-022-01262-0#Sec2
## Raised Questions:
Legitemacy of the relation between bvp and ecg?
GSR and its relation to the bioimpedance?
GSR,EDA,SKT,SCL and the relation of all?
Transformation of the ppg into either ECG and BVP?
Variation in sensors in higher end watches than the ppg sensor?
Tizen:https://docs.tizen.org/application/native/guides/location-sensors/device-sensors/#heart-rate-monitor-led-green-sensor for the ppg values

# BIOVID
## Brief:
A huge database of 94 participants watching a set of 3 movie clips for each 
discrete emotion the most important thing is that the classification has been done
on 5 discrete emotions(amusement,sad,angry,fear,disgust) where the biosignals taken
was taken from the video which achieved the required emotion by the highest degree

## pros:
A high range of age starting from 18 till 65 with a relative 50/50 male to female ratio
the exact numbers mentioned in the paper and there is a plethora of data in the database
any exact numbers or further info could be found here:
https://www.researchgate.net/publication/309779509_BioVid_Emo_DB_A_Multimodal_Database_for_Emotion_Analyses_validated_by_Subjective_Ratings
## cons:
-Although each video was taken with the highest associativity the final emotion isn't certainly
discrete where the association of the amusement clips for example was 7/10 and 1.63 in sadness
and 1.28 in anger and 1.09 in disgust and 1.22 in fear and this was the general analysis couldnt
be measuered on each sample as well as these values are very close when it comes to anger and
sadness.
-As well as the discoordinated of the video lengths its not the same across the whole dataset
where it starts from 32 to 245 second where we could take a snip of a continous 32 secs of each
clip to standardize our data


# probable datasets
## 1-ASCERTAIN Dataset(need accesss):
  58 users,36 clip
  each user watch stimuli clip and this data recorded alongside the video
  recorded data: ECG,GSR,EEG,Facial landmark trajectories
## user data annotated
    Personality scores for 5 Personality traits: Extraversion, Agreeableness, Conscientiousness, Emotional Stabily, Openness
    self report:Arousal, Valence, Engagement, Liking, Familiarity
## ASCERTAIN paper (need access for full pdf):
  https://ieeexplore.ieee.org/abstract/document/7736040
  they focus on Emotion and Personality Recognition using physiological signals mentioned above ,study physiological correlation of emotion and personality(not fully     understood ,don't know if it's helpful or not yet)
## 2-Mahinob needs access:
  ECG, EEG (32 channels), respiration amplitude, and skin temperature
  https://mahnob-db.eu/hci-tagging/
  any more data about the sampling method or the values itself could be found after gaining access
## 3-Stress datasets we dont know how would they be useful:
WESAD:https://cdn.discordapp.com/attachments/681869328466837514/1026668931341099050/unknown.png
SWEll:https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624/tab/2
(best because it applies a reallife situation)
## 4-Biraffe
  ECG,EDA,GSR
  useful:signals timestamped so could be trained using the VA
  issues:the null values of the EDA,personality traits affect the emotion recognition while it isnt used in the rest
# References(models and data understanding)
emognition.pwr.edu.pl/home
