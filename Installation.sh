git clone https://github.com/AlexEMG/DeepLabCut.git
# Install Git is you don't have one
# https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
cd DeepLabCut/conda-environments

# Anaconda must be installed beforehand
# When installing Anaconda, must sure to select "Add Anaconda to your PATH environment variable"
# Troubleshooting: https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10
conda env create -f dlc-windowsCPU.yaml
conda activate dlc-windowsCPU
pip install deeplabcut
pip install -U wxPython
pip install --ignore-installed tensorflow==1.10

# You may deactivate the environment by running    conda deactivate dlc-windowsCPU   after jobs done
# In case you would like to delete the entire environment (all the installed package 
# inside this envir will be removed from workstation): conda remove -n dlc-windowsCPU -all
