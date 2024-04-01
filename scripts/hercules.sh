#!/bin/bash
for DB in `ls ./results/`
do
    echo $DB
    
    for experiment in `ls ./results/${DB}/`
    do
        echo $experiment
        # Si el experimento no es Baseline 
        if [ $experiment != "Baseline_efficientnet_v2_s" ]
        then
            # Ejecuta el script revel_metrics.sh
            sbatch -o ./output/${DB}_${experiment}.txt \
                -J $DB$experiment \
                -c 1 \
                -x node[01-8] \
                ./scripts/revel_metrics.sh ./results/${DB}/${experiment}/ &
        fi
    done
    
done

# Imprime el número de jobs que se están ejecutando
echo "Number of jobs: $(squeue -u $USER | wc -l)"
