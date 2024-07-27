#!/bin/bash

# Submit the p1 job and capture its job ID
job_id=$(./p1_slurm.sh $1 $2 $3 DS_Sizes.csv $1 | sbatch | awk '{print $NF}')

# Submit the p2 job with a dependency on the p1 job
./p2_slurm.sh $1 $2 $3 DS_Sizes.csv $1 | sbatch --dependency=afterok:$job_id
