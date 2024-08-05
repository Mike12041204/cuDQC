#!/bin/bash

# Function to generate and print a job script
print_job_script() {
    local output=$1

    cat <<EOT
#!/bin/bash
#SBATCH --job-name=DcuQC_${output}_p3           # set job name
#SBATCH --partition=amperenodes                 # set job partition
#SBATCH --time=12:00:00                         # set job max time
#SBATCH --output=o_${output}_p3.txt             # set output path for nodes
#SBATCH --error=e_${output}_p3.txt              # set error path for nodes
#SBATCH --nodes=1                               # 4 nodes
#SBATCH --ntasks-per-node=1                     # 1 process per node
#SBATCH --cpus-per-task=1                       # 1 thread per process
#SBATCH --mem-per-cpu=80G                       # 80GB memory per thread
#SBATCH --gres=gpu:1                            # 1 GPU per node

# Load the MPI module
module load OpenMPI/4.1.5-GCC-12.3.0
module load CUDA/12.2.0

srun --mpi=pmix_v3 ./program3 ${output}
EOT
}

# Generate and print the job script with the provided arguments
print_job_script $1