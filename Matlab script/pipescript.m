% MAKE SURE TO HAVE THE CURRENT PATH SAME AS THIS SCRIPT'S

clear;
gcp;
path = pwd;
input= 'E:\A RSync FungWongLabs\Raw Data\Witnessing\female\Round 8\3_22_2019\H9_M22_S38';
mouse_id='1053';
session_type='SI_B';
hour='9_22_38';
mergedVideos=pipejoinavi(input,mouse_id, session_type, hour);
% Change the path to NoRMCorre in order to get functions%
[Mr,tiff_name]=pipenormcorre(mergedVideos);
saveastiff(Mr,tiff_name);
cnmfeRun(tiff_name);
del_files(tiff_name);


