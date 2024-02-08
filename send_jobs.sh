#!/bin/bash

for dataset in "CIFAR10" "CIFAR100" "EMNIST" "FashionMNIST" "Flowers" "OxfordIIITPets"
do
    sbatch --job-name=${dataset} --output=./outputs/${dataset}.out train_slurm.sh $dataset
done
