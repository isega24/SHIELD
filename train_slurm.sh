#!/bin/bash

# Nombre del proceso
#SBATCH --partition dgx,dgx2 # Cola para ejecutar (dgx o dios)
#SBATCH --gres=gpu:1 # Numero de gpus a usar
#SBATCH --mem=32G # Memoria a usar


export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate ./.conda

for percentage in 15 18 21 24 27
do
    for model in "efficientnet-b2" "efficientnet_v2_s"
    do
        # Baseline
        python -u SHIELD/tests/train.py --dataset $1 --pretrained_model $model --xshield --percentage $percentage
        python -u SHIELD/tests/train.py --dataset $1 --pretrained_model $model --shield --percentage $percentage
    done
done