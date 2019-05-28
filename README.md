# Neural-Decoding-Project
Hippocampus spike activity related to the depression-related behaviors after stress


## Correlation

Pearson correlation coefficient, Python code with Matlab compatible file.  

[Reference](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

## Deeplabcut Tutorial

Installation and usage of DeepLabCut

[Reference](https://github.com/AlexEMG/DeepLabCut)

## Some Matlab scripts

* Extrat csv from binarized Mat.m
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
