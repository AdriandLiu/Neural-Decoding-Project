# Stress Determination Research Project

This project aims to use the state-of-the-art ML/DL model to determine the stress level of the mouse based on its hippocampus neuron activity and motor activity. We thus propose a ***Stress Determinator*** to apply into neuron decoding regarding the mouse stress level.

Hippocampus neuron activities are recorded by millisecond camera and processed by CNMF-E, and motor activity are recorded by camera and processed by DeepLabCut. Tutorials for both toolboxs are available in this repo. 

Then use the timestamp of motor and millisecond cameras to align the frames between them. Finally feed data into ML/DL model. Tutorial available in preProcessor_alignNeuronBehav.ipynb.

ML/DL model source code is available in src and utils.

**Data used:**

All the data sourced from Dr. Wong's mouse neuron experiments in Douglas Research Center with two major types of data: neuron and behavior data, and two main categories used in our model training experiments:

* Bullying mouse in the enclosure

<p align="center"><img src="https://github.com/Adriandliu/Neural-Decoding-Project/blob/master/img/one_free.PNG?raw=true" width="200"/>
</p>


* Bullying and defeated mice are both free to move in the cage


<p align="center"><img src="https://github.com/Adriandliu/Neural-Decoding-Project/blob/master/img/two_free.PNG?raw=true" width="150"/>
</p>

**Important:**

In order to make this experiment reproducible, please make sure the following data are available for input:
* Neuron activity data, primarily extracted from [CNMF-E](https://github.com/zhoupc/CNMF_E), available in CNMF-E folder
* Mouse behavioral data, primarily extracted and labeled from [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), available in DeepLabCut folder
* Timestamp file that automatically generated by the camera and its application is file for aligning behavioral camera and neuron camera, available in the index format of mouse experiments date and time (./Raw data).


## Preprocessing

We preprocess the neuron and motor behavior data by aligning their video frames based on the recorded timestamp file. This is the first step to enable the running of the **Stress Determinator**

The tutorials are available [here](https://github.com/AdriandLiu/Neural-Decoding-Project/blob/master/utils/preprocessing.ipynb) and [here](https://github.com/AdriandLiu/Neural-Decoding-Project/blob/master/utils/preProcessor_alignNeuronBehav.ipynb)

## Stress Determinator

The source code and tutorial are available [here](https://github.com/AdriandLiu/Neural-Decoding-Project/tree/master/src), including the Bi-directional LSTM/GRU and Transformer, which is the **Stress Determinator** that we proposed. 

## Presentation

A summary report and presentation available [here](https://github.com/AdriandLiu/Neural-Decoding-Project/tree/master/reports), including the brief research objectives, procedures and experiments results.


</br>
</br>
</br>
</br>
</br>
</br>

The followings are the techniques and toolboxs that we used and how the data was prepared. 


## Correlation	

Pearson correlation coefficient, Python code with Matlab compatible file.  	


[Reference](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)	


## Deeplabcut Tutorial	

[Reference](https://github.com/AlexEMG/DeepLabCut)	

## Some Matlab scripts	

* Extrat csv from binarized Mat.m	This is the notebook for aligning neuron and behavior data by their timestamp. Used as a preprocessor to clean and prepare the integrated data for DL/ML models.
* Three-DPlotForMr.m	
* avitotif.m	
* joinavi_gs.m	
* saveastiff.m	

## Neuron Detection	

[CNMF-E](https://github.com/zhoupc/CNMF_E) algorithm application, detects neurons based on their luminance. 	

joinavi_gs.m -> demo-1p-low-RAW.m -> saveastiff.m -> cnmfe-setup.m -> demo-large-data-1p.m	

  File name    | Functionality           |	
|--------------|-------------------------|	
| joinavi.m    |   - Merge all the msCam videos|	
| demo-1p-low-RAW.m   |   - Convert videos to data array|	
| saveastif.m              |           - Convert data array to tiff file|	
| cnmfe-setup.m           |       - Setup the environment for using cnmfe|	
| demo-large-data-1p.m    | - Neuron detection|	

## Association rules	

Using Data Mining technique - Association rules to find out the relationship between different neurons. 	

## Preprocessing_behav	

* Read coordiate date	
* Distance between defeated mouse head and encloser	
* Defeated mouse head direction	
* Defeated mouse behavior annotations	
* Appendix: video rotation



# Readme for preprocessing 

## Setup
**!!! Remember to put all the source codes from https://github.com/zhoupc/CNMF_E in the same path of *cnmfe_setup.m* before running it**
1. Add the pipeline folder and all its subfolders to matlab path.
1. If this is the first time, run *cnmfe_setup.m* to set up environment.
1. The **input** variable in *pipescript.m* should be a char array with they path to the folder with all the *.avi* files to be converted.
1. The **mouse_id** variable in *pipescript.m* should be a char array with the mouse id.
1. The **session_type** variable in *pipescript.m* should be a char array of the type of session being recorded (hab,def1, ...).
1. The **hour** variable in *pipescript.m* should be a char array of the time associated with this session. It should be formated as hour\_minute\_second.
1. The **cnmfe_home** variable in *pipejoin.m* should be a char array of the path where the cnmfe analysis should be outputted. This will also be where the merged *.avi* file and the *.tif* file are generated.
1. Run with by running *pipescript* on the matlab command window.




## Errors
* This code does not work with behavCam videos. It has only been tested with msCam videos.
* All the input values should be in the form of char arrays. NO STRINGS. This will lead to an error when the code runs *saveastif.m*.
* Some files are very large and will take a long time to process. for these files, if the connection to the external drive is broken, then the output will be a broken tiff file. The error you will get will be **fl:filesystem:SystemError**. All you need to do is re-establish the connection and re-run the preprocessing algorithm.

## Misc
* If you wish to supress or delete the *.avi*  file, remove the percent symbols in *pipenormcorre.m* on lines 52 
* To delete the files, run *del_files.m* with **tiff_name** as input

## Legend for the readme
* Filenames are in *italics*.
* Variables are in **bold**.

## preProcessor_alignNeuronBehav.ipynb

This is the notebook for aligning neuron and behavior data by their timestamp. Used as a preprocessor to clean and prepare the integrated data for DL/ML models.
