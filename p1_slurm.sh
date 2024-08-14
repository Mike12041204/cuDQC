#!/bin/bash

# Function to generate and print a job script
print_job_script() {
    local graph=$1
    local gamma=$2
    local min_size=$3
    local dssizes=$4
    local output=$5

    cat <<EOT
#!/bin/bash
#SBATCH --job-name=DcuQC_${output}_p1		    # set job name
#SBATCH --partition=amd-hdr100                  # set job partition
#SBATCH --time=12:00:00                         # set job max time
#SBATCH --output=o_${output}_p1.txt	            # set output path for nodes
#SBATCH --error=e_${output}_p1.txt      	    # set error path for nodes
#SBATCH --nodes=1                               # 1 nodes
#SBATCH --ntasks-per-node=1                     # 1 process per node
#SBATCH --cpus-per-task=1                       # 1 thread per process
#SBATCH --mem-per-cpu=80G                       # 80GB memory per thread
#SBATCH --gres=gpu:0                            # 0 GPU per node

# Load the MPI module
module load OpenMPI/4.1.5-GCC-12.3.0
module load CUDA/12.2.0

srun --mpi=pmix_v3 ./program1 ../graph_files_cuQC/${graph} ${gamma} ${min_size} ${dssizes} ${output}
EOT
}

# Generate and print the job script with the provided arguments
print_job_script $1 $2 $3 $4 $5