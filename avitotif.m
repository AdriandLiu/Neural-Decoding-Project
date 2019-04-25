function [] = avitotif(pathToMovie)
obj = VideoReader(pathToMovie);
vid = read(obj);
frames = obj.NumberOfFrames;
[filepath,name,ext] = fileparts(pathToMovie);
for x = 1 : frames
    imwrite(vid(:,:,:,x),strcat('frame-',name,'.tif'),'WriteMode','append');
end