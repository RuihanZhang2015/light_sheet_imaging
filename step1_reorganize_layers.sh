#!/bin/bash                      
#SBATCH -t 24:00:00          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB          
#SBATCH --constraint=rocky8
source ~/.bash_profile
conda activate zeguan
python -u /om2/user/zgwang/light_sheet_imaging/step1_reorganize_layers.py