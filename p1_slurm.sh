#!/bin/bash

# Function to generate and print a job script
print_job_script() {
    local graph=$1
    local out_gamma=$2
    local in_gamma=$3
    local min_size=$4
    local dssizes=$5
    local output=$6

    cat <<EOT
#!/bin/bash
#SBATCH --job-name=DQC-P1Z_${output} 		    # set job name
#SBATCH --partition=amd-hdr100                  # set job partition
#SBATCH --time=12:00:00                         # set job max time
#SBATCH --output=DQC-O1Z_${output}	            # set output path for nodes
#SBATCH --error=DQC-E1Z_${output}        	    # set error path for nodes
#SBATCH --nodes=1                               # 1 nodes
#SBATCH --ntasks-per-node=1                     # 1 process per node
#SBATCH --cpus-per-task=1                       # 1 thread per process
#SBATCH --mem-per-cpu=80G                       # 80GB memory per thread
#SBATCH --gres=gpu:0                            # 0 GPU per node

# Load the MPI module
module load OpenMPI/4.1.5-GCC-12.3.0
module load CUDA/12.2.0

srun --mpi=pmix_v3 ./program1 ${graph} ${out_gamma} ${in_gamma} ${min_size} ${dssizes} ${output}
EOT
}

# Generate and print the job script with the provided arguments
print_job_script $1 $2 $3 $4 $5 $6