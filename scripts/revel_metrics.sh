#!/bin/bash
#SBATCH --partition muylarga

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate /home/isevillano/Github/SHIELD/.conda

python -u SHIELD/tests/REVEL_metrics.py \
    --run_fold $1