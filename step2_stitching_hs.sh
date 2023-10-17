#!/bin/bash                      
#SBATCH -t 24:00:00          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB          
#SBATCH --constraint=rocky8
source ~/.bash_profile
source /etc/profile.d/modules.sh
conda activate voltage
python -u /om2/user/zgwang/light_sheet_imaging/step2_stitching_hs.py