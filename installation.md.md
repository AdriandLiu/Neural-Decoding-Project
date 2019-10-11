# Protocol for DeepLabCut (DLC) Installation

This protocol details the steps to install DLC with the acceleration from graphics cards in Windows 10. NVIDIA cards are required. Neither AMD nor Intel cards support that feature.

## Install Graphics Card Driver

1.	Download the latest version of the driver for your video card from the NVIDIA website (https://www.nvidia.com/Download/index.aspx).
2.	Install the driver. Choose the ‘Express (Recommended)’ installation option when asked.
3.	Do not need to install CUDA or cuDNN manually. They will be automatically installed during the creation of the DLC environment.  It is still fine if you have accidentally installed any of them since they will be ignored.
4.	NVIDIA drivers seem perfectly compatible with lower versions of CUDA and cuDNN. Thus, just try the latest one before looking into the complicated compatibility table (https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html).

## Install Git

1.	Download Git from its official website (https://git-scm.com/downloads).
2.	Install Git. Choose the default option whenever it asks.

## Install Anaconda

1.	Make sure that the current Windows user is an administrator. If not, follow the steps listed on https://www.windowscentral.com/how-change-user-account-type-windows-10 to change the user into an administrator.
2.	Download Anaconda from its official website (https://www.anaconda.com/distribution/). Choose the Python 3.7 version.
3.	When the setup asks ‘Install for:’, choose ‘Just Me (recommended)’.
4.	Untick the advanced option to ‘Add Anaconda to my PATH environment variable’.

## Download DLC and Create DLC Environment

1.	Open Git Bash. Run ‘git clone https://github.com/AlexEMG/DeepLabCut’.
2.	Open Anaconda Powershell Prompt. Run ‘cd DeepLabCut/conda-environments’.
3.	Run ‘conda env create -f dlc-windowsGPU.yaml’.
Open Jupyter Notebook in DLC Environment
1.	Run ‘conda activate dlc-windowsGPU’.
2.	Run ‘jupyter notebook’ followed by the path of the .ipynb file.

## Reinstallation

1.	When DLC crashes, it is suggested to remove and reinstall the graphics card driver and Anaconda before reinstalling DLC.
2.	Go to Windows Settings -> Apps -> Apps & features to remove all apps with the word NVIDIA or Anaconda.
3.	Delete the ‘.conda’, ‘.ipython’, ‘.jupyter’, ‘.keras’, ‘.matplotlib’, and ‘Anaconda3’ folders in the Windows current user’s home directory if they have not been deleted.
4.	Delete the ‘DeepLabCut’ folder in the user’s home directory.
5.	Follow the steps given to reinstall the video card driver and Anaconda and download DLC and create DLC environment.

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTQyMTI0ODA4XX0=
-->