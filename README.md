# PFDGenerator

## Environment Setting
1. first, download Anaconda3 (our model is based on Anaconda3-2023.03-1-Linux-x86_64.sh)
2. input below statement to set conda environment:

  conda env create -f environment.yml

3. activate gpt2_env, run below files for training:
'''
conda activate gpt2_env
python pretrain.py
python finetune.py
'''
4. activate sfiles_env, run below files for generating:
'''
conda activate sfiles_env
python print.py
'''

you can edit 'print.py' to set the input string.
