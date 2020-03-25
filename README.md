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
