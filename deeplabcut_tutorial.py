import deeplabcut
import os
import numpy as np
import cv2
import glob
from os import listdir
from os.path import isfile, join








class tutorial():
    def __init__(self, algo, crop, userfeedback, shuffle, saveiters, displayiters, angle, center, scale):
        self.algo = "automatic"
        self.crop = False
        self.userfeedback = False
        self.shuffle = 1
        self.saveiters = 200
        self.displayiters = 100
        self.angle = -4.5
        self.center = None
        self.scale = 1.0

    def rotate(self, image, angle, center=None, scale=1):
    #scale = 1: original size
    rows,cols,ch = image.shape
    if center == None:
        center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    #Matrix: Rotate with center by angles
    dst = cv2.warpAffine(image,M,(cols,rows))
    #After rotation
    return dst


    def videorotate(filename, output_name, display_video = False):
        # capture video
        cap = cv2.VideoCapture(filename)

        #read video frame by frame
        #extract original video frame features
        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        #Make a directory to store the rotated videos
        path = "./rotated"
        try:
            os.mkdir(path)
        except OSError:
            pass
        else:
            print ("Successfully created the directory %s " % path)

        #Automatically name the rotated videos
        file = "./rotated/" + output_name
        out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, sz)
        #Integrate all frames to video


        #Read videos and rotate by certain degrees
        while(cap.isOpened()):
            #flip for truning(fliping) frames of video
            ret,img = cap.read()
            try:
                img2 = rotate(img, -4.5)
                #Flipped Vertically
                out.write(img2)
                if display_video == True:
                    cv2.imshow('rotated video',img2)

                k=cv2.waitKey(30) & 0xff
                #once you inter Esc capturing will stop
                if k==27:
                    break
            except:
                print (filename, 'successfully rotated!!!' )
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    #Generate all rotating videos
    filenames = glob.glob('*.mp4') #Return the file name with .mp4 extention
    for i in filenames:
        videorotate(i,os.path.splitext(i)[0] + " rotated.mp4")


    cwd = os.chdir("./rotated")
    #we are using rotated videos
    cwd = os.getcwd()
    mp4files = [f for f in listdir(cwd) if isfile(join(cwd, f)) and os.path.splitext(f)[1] == ".mp4"]
    #Get all mp4 files

    task='Reaching' # Enter the name of your experiment Task
    experimenter='Donghan' # Enter the name of the experimenter
    video=mp4files # Enter the paths of your videos you want to grab frames from.

    path_config_file=deeplabcut.create_new_project(task,experimenter,video, working_directory='/home/donghan/DeepLabCut/data/rotated',copy_videos=True)
    #change the working directory to where you want the folders created.


# The function returns the path, where your project is.
# You could also enter this manually (e.g. if the project is already created and you want to pick up, where you stopped...)
#path_config_file = '/home/Mackenzie/Reaching/config.yaml' # Enter the path of the config file that was just created from the above step (check the folder)

    %matplotlib inline
    deeplabcut.extract_frames(path_config_file,'automatic',crop=False, userfeedback=False) #there are other ways to grab frames, such as by clustering 'kmeans'; please see the paper.
    #You can change the cropping to false, then delete the checkcropping part!
    #userfeedback: ask if users would like to continue or stop

    %gui wx
    deeplabcut.label_frames(path_config_file)

    deeplabcut.check_labels(path_config_file) #this creates a subdirectory with the frames + your labels

    deeplabcut.create_training_dataset(path_config_file)

    deeplabcut.train_network(path_config_file, shuffle=1, saveiters=200, displayiters=10)
#Other parameters include trainingsetindex=0,gputouse=None,max_snapshots_to_keep=5,autotune=False,maxiters=None
#Detailed function explanation can be found here https://github.com/AlexEMG/DeepLabCut/blob/efa95129061b1ba1535f7361fe76e9267568a156/deeplabcut/pose_estimation_tensorflow/training.py

    deeplabcut.evaluate_network(path_config_file)

    videofile_path = ['1035 SI_A, Aug 15, 13 17 7 rotated.mp4'] #Enter the list of videos to analyze.
    deeplabcut.analyze_videos(path_config_file,videofile_path)

    deeplabcut.create_labeled_video(path_config_file,videofile_path)
