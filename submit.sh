#!/bin/bash
#SBATCH --mem=8GB
#SBATCH --time=0-1
#SBATCH --mail-user=arturs.polis@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

# Print out the invocation parameters:
echo "Executing: ${0} ${@}"

module purge
module load anaconda3
module list

# Launch python script
srun python3.6 $*
