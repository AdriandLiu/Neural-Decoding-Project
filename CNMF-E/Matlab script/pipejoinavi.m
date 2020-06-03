function res = pipejoinavi(path,mouse_id,session_type,hour)

previouspath = pwd;
cd(path);
addpath(path);
tmp = dir('msCam*.avi');        % all msCam .avi video clips
videoList = {tmp.name}';   % sort this list if necessary! sort(videoList) might work
cd(previouspath);
videoList=filesort(videoList);

%using the location of cnmfe output to keep the results 
cnmfe_home='\\Dmas-ws2017-006\h\RSync FungWongLabs\CNMF-E\';
mouse_folder=strcat(cnmfe_home,mouse_id);       %making the mouse folder
mkdir(mouse_folder);
type_folder=strcat(mouse_folder,'\', session_type);    %making the session type folder
%mkdir(type_folder);
hour_folder=strcat(type_folder,'\',hour);
mkdir(hour_folder);
cd(hour_folder);
name=strcat(mouse_id,'_',session_type,'.avi');

% create output in seperate folder (to avoid accidentally using it as input)
outputVideo = VideoWriter(fullfile(hour_folder,name),'grayscale AVI');
% if all clips are from the same source/have the same specifications
% just initialize with the settings of the first video in videoList
inputVideo_init = VideoReader(videoList{1}); % first video
outputVideo.FrameRate = inputVideo_init.FrameRate;
tic
open(outputVideo) % >> open stream
% iterate over all videos you want to merge (e.g. in videoList)
for i = 1:length(videoList)
    % select i-th clip (assumes they are in order in this list!)
    inputVideo = VideoReader(videoList{i});
    % -- stream your inputVideo into an outputVideo
    while hasFrame(inputVideo)
        writeVideo(outputVideo, readFrame(inputVideo));
    end
    fprintf('\n %u done out of %u ',i,length(videoList));
end
toc
close(outputVideo) % << close after having iterated through all videos
res= fullfile(hour_folder,'\',name);
