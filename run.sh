#!/bin/bash

filepath="../graph_files_cuDQC/"

job_id0=$(./p0_slurm.sh ${filepath} $1 $2 $3 $4 | sbatch | awk '{print $NF}')

# # Submit the p1 job and capture its job ID
# job_id1=$(./p1_slurm.sh ${filepath}$1 $2 $3 DS_Sizes.csv $1 | sbatch --dependency=afterok:$job_id0 | awk '{print $NF}')

# # Submit the p2 job with a dependency on the p1 job
# job_id2=$(./p2_slurm.sh ${filepath}$1 $2 $3 DS_Sizes.csv $1 | sbatch --dependency=afterok:$job_id1 | awk '{print $NF}')

# ./p3_slurm.sh $1 | sbatch --dependency=afterok:$job_id2