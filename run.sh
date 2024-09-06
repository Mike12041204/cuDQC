#!/bin/bash

filepath="../graph_files_cuDQC/"
output="$1-$2-$3-$4"

./slurm.sh ${filepath}$1 $2 $3 $4 DS_Sizes.csv ${output} | sbatch