emognition : 
	
	.participants : 43, (21f, 22m), aged(19, 29), polish
	
	.signal : 
		-physiological : EEG, BVP, HR, EDA, SKT, ACC, GYRO(in parallel)
		-with upper body videos(facial expression)
	
	.self reports: 1-discrete emotions, 2-VAM(motivation)

	.target emotions : amusement, awe, enthusiasm, liking, surprise, digust, fear, sadness
	
	.devices and signals : 	
		-Muse 2 equipped with electroencephalograph (EEG), accelerometer (ACC), and gyroscope (GYRO) sensors
		-Empatica E4 measuring and providing blood volume pulse (BVP), electrodermal activity (EDA), skin temperature (SKT), and also providing interbeat interval (IBI), and ACC data
		-Samsung Galaxy Watch SM-R810 measuring and providing BVP, and also providing heart rate (HR), peak-to-peak interval (PPI), ACC, GYRO, and rotation data
	
	.stimuli : 
		audio video (films clips){https://www.tandfonline.com/doi/abs/10.1080/02699930903274322}, 10 short movies (9 emotions + neutral)

	.measure(self assessment):
		After each movie, participants filled out two questionnaires:
			1. Nine categorical emotions (enthusiasm, liking, sadness, anger, disgust, surprise, fear, amusement, and awe) with a 5-point Likert scale
			2. Three-dimensional SAM (valence, arousal, motivation) on a 9-point Likert scale

	#Empatica E4
	Signals are explained by the producer at:
	https://support.empatica.com/hc/en-us/sections/200582445-E4-wristband-data
		* blood volume pulse (BVP;  64Hz)
		* interbeat interval (IBI, sampling rate variable)
		* electrodermal activity (EDA; 4Hz)
		* 3-axis accelerometer (ACC - X, Y, Z axes; 32Hz)
		* skin temperature (TEMP; 4Hz)

	#Samsung Galaxy Watch SM-R810
	Signals were obtained with a custom made Tizen application. The obtained signals are explained in the documentation: https://docs.tizen.org/application/native/guides/location-sensors/device-sensors 
		* heart rate (heartRate; 10Hz - 100ms)
		* Peak-to-peak interval (PPInterval; 10Hz - 100ms)
		* raw blood volume pulse (BVPRaw; 20Hz - 50ms)
		* processed blood volume pulse (BVPProcessed; 20Hz - 50ms)
		* 3-axis accelerometer (acc - X, Y, Z axes; ~33Hz - 30ms)
		* 3-axis gyroscope (gyr - X, Y, Z axes; ~33Hz - 30ms)
		* 4-axis rotation (rot - X, Y, Z, W axes; ~33Hz - 30ms)

	#study_data.zip contains physiological signals, self-reports, and control questionnaires;-->quantum(may be use for six basic emotions, annotated by probability for each emotion)
	
	#{participant_id}_{emotion}_{experiment_stage}_{device}.json
		e.g. 22_FEAR_STIMULUS_MUSE.json
	
	#Questionnaires and metadata:
	{participant_id}_QUESTIONNAIRES.json
	e.g. 22_QUESTIONNAIRES.json
	
	#Annotations (OpenFace and Quantum):
	{participant_id}_{emotion}_{experiment_stage}.json/csv
	e.g. 22_FEAR_STIMULUS.json/csv
	
	#Possible values:
	participant_id: integers from 22 to 64
	emotion: BASELINE, NEUTRAL, AWE, DISGUST, SURPRISE, ANGER, ENTHUSIASM, LIKING, FEAR, AMUSEMENT, SADNESS
	experiment_stage: WASHOUT, STIMULUS, QUESTIONNAIRES
	device: EMPATICA, SAMSUNG_WATCH, MUSE


	#study_data.zip
	{participant_id}_{emotion}_{experiment_stage}_{device}.json
	e.g. 22_FEAR_STIMULUS_MUSE.json


	Participant’s folder contains signal data and questionnaires data.
	All physiological and questionnaire data are in JSON format. Each file contains a JSON object with signal types and JSON arrays of signal values.
	Because of variable sampling rates of sensors, Empatica E4 and Samsung Galaxy Watch SM-R810 data are tuples of timestamp and signal value.

	#{participant_id}_QUESTIONNAIRES.json
	e.g. 22_QUESTIONNAIRES.json


	Questionnaire files contain a JSON object with anonymized participant’s metadata and a JSON array of questionnaires filled out during the examination.


	Metadata include:
	id, age, gender, info about wearing glasses [true/false], time from wake up [hh:mm], time from activity [hh:mm], time from caffeine [hh:mm], time from last cigarette [hh:mm], time from last meal [hh:mm], other drugs in last eight hours [true/false], weekday of the experiment, experiment start date and time, experiment end date and time, a list of movies used in the experiment that have been seen already by the participant, the order of the movies in the experiment.


	Questionnaire data include:
	consumed stimulus, questionnaire start and end date and time, answers from both self-assessments

	#openFace (may be useful if we use modal for eye gaze tracking)

	#Quantum.zip
	{participant_id}_{emotion}_{experiment_stage}.json
	e.g. 22_FEAR_STIMULUS.json


	The files contain basic emotions recognized from downscaled (640x360, 60 fps) facial recordings using Quantum Sense software (Quantum CX, Poland). The software outputs the probability that the following emotions occurred in a given frame: anger, disgust, happiness, sadness, surprise, neutral. Additionally, head pose is provided (X, Y, Z, yaw, pitch, roll). Each frame has a timestamp in milliseconds, which corresponds to a time of Quantum Sense processing (not the time of the recording).

	#data_completeness.csv


	This file contains information about missing data/files due to the researcher forgetting to start the recording, device malfunction, or other reasons.


	For the columns: ‘Empatica E4’, ‘Muse 2’ and ‘Samsung Watch’ a single row corresponds to all phases of the stimulus (washout + movie + self-assessment) or baseline (movie + self-assessment).
	Values in the ‘Muse 2’ column represent the percentage of the time when a headband was on the head according to the sensor, i.e., the HeadBandOn value provided by the device. Value can be from 0 (electrodes not in contact with the head throughout the particular film condition) to 1 (contact over 100% time). Lower values may indicate a low quality or corrupted signal.


	For the columns: ‘Facial video’, ‘Pose’ and ‘Head pitch’ a single row corresponds only to the stimulus movie phase.
		Values in the ‘Facial video’ column contain the percentage of frames where at least one face was detected, according to data from Quantum Sense software.
		Values in the ‘Pose’ column indicate whether the participant was leaning towards the camera or sitting straight during the majority part of the movie.
		Values in the ‘Head pitch’ column contain the mean pitch angle returned by Quantum Sense software.


	Furthermore, some rows include notes in the ‘Comments’ column. ‘Skip’ denotes that the participant decided to omit part of the stimulus. ‘Delay’ denotes that the stimulus phase took longer than expected, i.e., the application needed some time to load the questionnaires. Additionally, in each such case, the time of skip/delay is provided. ‘Seen’ denotes that the participant has already seen the movie before the study.
 
 .assumpitons of experiment:
	The exclusion criteria were significant health problems, use of drugs and medications that might affect cardiovascular function, prior diagnosis of cardiovascular disease, hypertension, or BMI over 30 (classified as obesity). 
	We asked participants to reschedule if they experienced an illness or a major negative life event.
	The participants were requested (1) not to drink alcohol and not to take psychoactive drugs 24 hours before the study 
	(2) to refrain from caffeine, smoking, and taking nonprescription medications for two hours before the study 
	(3) to avoid vigorous exercise and eating an hour before the study. 
	Such measures were undertaken to eliminate factors that could affect cardiovascular function

 .important: 
	---->he obtained data facilitates various ER approaches, e.g., multimodal ER, EEG- vs. cardiovascular-based ER, discrete to dimensional representation transitions. 
	The technical validation indicated that watching film clips elicited the targeted emotions.It also supported signals’ high quality
	
	--->The Emognition dataset offers the following advantages over the previous datasets(POPANE, DEAP, DECAF, BIRAFFE, MAHNOB-HCI, ASCERTAIN, QAMAF):
		(1) the physiological signals have been recorded using wearables which can be applied unobtrusively in everyday life scenarios 
		(2) the emotional state has been represented with two types of emotional models, i.e., discrete and dimensional 
		(3) nine distinct emotions were reported 
		(4) we put an emphasis on the differentiation between positive emotions; thus, this is the only dataset featuring four discrete positive emotions; the differentiation is important because studies indicated that specific positive emotions might differ in their physiology
		(5) the dataset enables versatile analyses within emotion recognition (ER) from physiology and facial expressions

	--->The Emognition dataset may serve to tackle the research questions related to: 
		(1) multimodal approach to ER 
		(2) physiology-based ER vs. ER from facial expressions 	
		(3) ER from EEG vs. ER from BVP 
		(4) ER with Empatica E4 vs. ER using Samsung Watch (both providing BVP signal collected in parallel) 
		(5) classification of positive vs. negative emotions 
		(6) affect recognition – low vs. high arousal and valence 
		(7) analyses between discrete and dimensional models of emotions

	--->For the baseline, participants watched dots and lines on a black screen for 5 minutes (physiological baseline) and reported current emotions (emotional baseline) using discrete and dimensional measures. The main part of the experiment consisted of ten iterations of 
		(1) a 2-minute washout clip (dots and lines) 
		(2) the emotional film clip 
		(3) two self-assessments

	--->The order of film clips was counterbalanced using a Latin square, i.e., we randomized clips for the first participant and then shifted by one film clip for each next participant so that the first film clip was placed as the last one
	
	--->The most common approach to emotion recognition from physiological signals includes 
		(1) data collection and cleaning 
		(2) signal preprocessing, synchronization, and integration 
		(3) feature extraction and selection  
		(4) machine learning model training and validation

 .data preprocessing:
	-The processing of BVP signal from the Samsung Watch PPG sensor consisted of subtracting the mean component, eight-level decomposition using Coiflet1 wavelet transform, and then reconstructing it by the inverse wavelet transform based only on the second and third levels. 
	 Amplitude fluctuations were reduced by dividing the middle value of the signal by the standard deviation of a one second long sliding window with an odd number of samples. The final step was signal normalization to the range of [−1,1].

 .data records:
	-The data are grouped by participants. Each participant has their folder containing files from all experimental stages (stimulus presentation, washout, self-assessment) and all devices (Muse 2, Empatica E4, Samsung Watch). In total, each participant has 97 files related to:

		10 film clips × 3 devices × 3 phases (washout, stimulus, self-assessment) = 90 files with signals

		baseline × 3 devices × 2 phases (baseline, self-assessment) = 6 files with signals

		a questionnaires.json file containing self-assessment responses, the control questionnaire, and some metadata (demographics and information about wearing glasses)

 .technical validation:
	-To test whether film clips elicit targeted emotions, we used repeated-measures analysis of variance (rmANOVA) with Greenhouse-Geisser correction and calculated recommended effect sizesfor ANOVA tests
	-To examine differences between the conditions (e.g., whether self-reported amusement in response to the amusing film clips was higher than it was reported in response to the other film clips), we calculated pairwise comparisons with Bonferroni correction of p-values for multiple comparisons
	-To validate the quality of the recorded physiological signals, we computed signal-to-noise ratios (SNRs) by fitting the second-order polynomial to the data obtained from the autocorrelation function. It was done separately for all physiological recordings (all participants, baselines, film clips, and experimental stages, see Sec. Data Records). SNR statistics indicated the signals’ high quality. Mean SNR ranged from 26.66 dB to 37.74 dB, with standard deviations from 2.27 dB to 11.13 dB. For one signal, the minimum SNR was 0.88 dB. However, 99.7% of its recordings had SNR values over 5.15 dB. As the experiments were conducted in a sitting position, we did not analyze signals from accelerometers and gyroscopes