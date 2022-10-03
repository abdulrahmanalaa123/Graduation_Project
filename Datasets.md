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

# probable datasets
##1-ASCERTAIN Dataset(need accesss):
  58 users,36 clip
  each user watch stimuli clip and this data recorded alongside the video
  recorded data: ECG,GSR,EEG,Facial landmark trajectories
## user data annotated
    Personality scores for 5 Personality traits: Extraversion, Agreeableness, Conscientiousness, Emotional Stabily, Openness
    self report:Arousal, Valence, Engagement, Liking, Familiarity
## ASCERTAIN paper (need access for full pdf):https://ieeexplore.ieee.org/abstract/document/7736040
    they focus on Emotion and Personality Recognition using physiological signals mentioned above ,study physiological correlation of emotion and personality(not fully understood ,don't know if it's helpful or not yet)
