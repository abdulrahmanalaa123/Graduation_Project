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
https://www.nature.com/articles/s41597-019-0209-0

## Modalities
ecg: data from the Electrocardiogram sensor.
bvp: data from the Blood Volume Pulse sensor.
gsr: data from the Galvanic Skin Response sensor.
rsp: data from the Respiration sensor.
skt: data from the Skin Temperature sensor.
emg_zygo: data from the Surface Electromyography (sEMG) sensor placed on the  Zygomaticus major muscles.
emg_coru: data from the Surface Electromyography (sEMG) sensor placed on the Corrugator supercilli muscles.
emg_trap: data from the Surface Electromyography (sEMG) sensor placed on the Trapezius muscles.

## Functional Analysis:
https://anson.ucdavis.edu/~mueller/Review151106.pdf

## useful because
interpolation method,realtime annotation(huge point),analysis method as well if understood,mapping if done(very probable its mentioned in the script sec)
## issues:
valence arousal mapping to discrete emotions (kinda referenced in the paper but not fully understood)

## anything else needed and the dataset itself can be found here:
https://gitlab.com/karan-shr/case_dataset/-/tree/master/
