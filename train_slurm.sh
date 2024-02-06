#!/bin/bash

#SBATCH --job-name JustOneProcess # Nombre del proceso
#SBATCH --partition dgx,dgx2 # Cola para ejecutar (dgx o dios)
#SBATCH --gres=gpu:1 # Numero de gpus a usar
#SBATCH --mem=32G # Memoria a usar


export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate ./.conda

python -u tests/train.py --dataset Flowers --pretrained_model efficientnet_v2_s --percentage 0 --batch_size 32
python -u tests/train.py --dataset Flowers --pretrained_model efficientnet_v2_s --shield --percentage 2 --batch_size 32

python -u tests/train.py --dataset OxfordIIITPet --pretrained_model efficientnet_v2_s --percentage 0 --batch_size 32
python -u tests/train.py --dataset OxfordIIITPet --pretrained_model efficientnet_v2_s --shield --percentage 2 --batch_size 32

python -u tests/train.py --dataset CIFAR10 --pretrained_model efficientnet_v2_s --percentage 0 --batch_size 32
python -u tests/train.py --dataset CIFAR10 --pretrained_model efficientnet_v2_s --shield --percentage 2 --batch_size 32

python -u tests/train.py --dataset CIFAR100 --pretrained_model efficientnet_v2_s --percentage 0 --batch_size 32
python -u tests/train.py --dataset CIFAR100 --pretrained_model efficientnet_v2_s --shield --percentage 2 --batch_size 32

python -u tests/train.py --dataset EMNIST --pretrained_model efficientnet_v2_s --percentage 0 --batch_size 32
python -u tests/train.py --dataset EMNIST --pretrained_model efficientnet_v2_s --shield --percentage 2 --batch_size 32

python -u tests/train.py --dataset FashionMNIST --pretrained_model efficientnet_v2_s --percentage 0 --batch_size 32
python -u tests/train.py --dataset FashionMNIST --pretrained_model efficientnet_v2_s --shield --percentage 2 --batch_size 32