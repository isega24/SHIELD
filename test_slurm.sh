#!/bin/bash

#SBATCH --job-name Testing # Nombre del proceso
#SBATCH --partition dgx # Cola para ejecutar (dgx o dios)
#SBATCH --gres=gpu:1 # Numero de gpus a usar
#SBATCH --output test_results.txt # Fichero de salida
#SBATCH --mem=16G # Memoria a usar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate ./.conda/
# If --force is used as an argument, it will delete the results of the previous run


# dataset tiene que ser el segundo argumento de la funcion
dataset=$2
# si no se especifica dataset, se ejecuta para todos los datasets
if [ -z "$dataset" ]
then
    echo "No dataset specified, running for all datasets"
    datasets="CIFAR10 CIFAR100 FashionMNIST EMNIST Flowers OxfordIIITPet"
fi
for dataset in $datasets;
do
    for file in `ls ./results/$dataset`;
    do
        echo $dataset $file
        if      [ "$1" == "--force" ]
        then
            echo "Forcing the test result"
            python -u SHIELD/tests/test.py --saved_dir ./results/$dataset/$file --force
        else
            python -u SHIELD/tests/test.py --saved_dir ./results/$dataset/$file
        fi
    done
done


