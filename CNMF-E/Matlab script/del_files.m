function [safe]= del_files(tiff_name)

[filepath,file_name,ext] = fileparts(tiff_name);
delete (fullfile(filepath,[file_name,'_filtered_data.h5']));
delete (fullfile(filepath,[file_name,'.avi']));
delete (fullfile(filepath,[file_name,'.tiff']));
safe='done';
