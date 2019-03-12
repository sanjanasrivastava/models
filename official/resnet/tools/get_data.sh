#!/bin/bash
#SBATCH -n 2
#SBATCH --array=0-5
#SBATCH --job-name=minimal
#SBATCH --mem=80GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om2/user/sanjanas/eccentricity-crop
singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om2/user/sanjanas/resnet-eccentricity/official/resnet/get_data.py  ${SLURM_ARRAY_TASK_ID}


