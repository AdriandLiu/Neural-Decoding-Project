git clone https://github.com/AlexEMG/DeepLabCut.git
cd DeepLabCut/conda-environments
conda env create -f dlc-windowsCPU.yaml
conda activate dlc-windowsCPU
pip install deeplabcut
pip install -U wxPython
pip install --ignore-installed tensorflow==1.10
