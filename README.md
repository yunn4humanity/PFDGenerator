# PFDGenerator

## Environment Setting
1. first, download Anaconda3 (our model is based on Anaconda3-2023.03-1-Linux-x86_64.sh)
2. input below statement to set conda environment:
<pre>
  <code>
    conda env create -f environment.yml
  </code>
</pre>


4. activate gpt2_env, run below files for training:
<pre>
  <code>
    conda activate gpt2_env
    python pretrain.py
    python finetune.py
  </code>
</pre>

5. activate sfiles_env, run below files for generating:
<pre>
  <code>
    conda activate sfiles_env
    python print.py
  </code>
</pre>

you can edit 'print.py' to set the input string.
