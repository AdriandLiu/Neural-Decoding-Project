cd('i:\video');
tmp = dir('*.avi');        % all .avi video clips
videoList = {tmp.name}';   % sort this list if necessary! sort(videoList) might work

% create output in seperate folder (to avoid accidentally using it as input)
mkdir('output');
outputVideo = VideoWriter(fullfile('i:\video','output/mergedVideo.avi'),'grayscale AVI');
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
end
toc
close(outputVideo) % << close after having iterated through all videos