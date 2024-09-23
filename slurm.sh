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
#SBATCH --job-name=DQC_${output} 		        # set job name
#SBATCH --partition=amperenodes                 # set job partition
#SBATCH --time=12:00:00                         # set job max time
#SBATCH --output=DQC-O_${output}	            # set output path for nodes
#SBATCH --error=DQC-E_${output}        	        # set error path for nodes
#SBATCH --nodes=4                               # 4 nodes
#SBATCH --ntasks-per-node=1                     # 1 process per node
#SBATCH --cpus-per-task=16                      # 16 thread per process
#SBATCH --mem=94G                               # 94GB memory per process
#SBATCH --gres=gpu:1                            # 1 GPU per process

# Load the MPI module
module load OpenMPI/4.1.5-GCC-12.3.0
module load CUDA/12.2.0

srun --mpi=pmix_v3 ./cuDQC ${graph} ${out_gamma} ${in_gamma} ${min_size} ${dssizes} ${output}
EOT
}

# Generate and print the job script with the provided arguments
print_job_script $1 $2 $3 $4 $5 $6