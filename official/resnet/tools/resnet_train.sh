#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=resnet
#SBATCH -t 12:00:00
#SBATCH --workdir=./log/
#SBATCH --mem=250GB
#SBATCH -c 80
#SBATCH --qos=cbmm


# export PYTHONPATH="$PYTHONPATH:/raid/poggio/home/xboix/src/robustness-imagenet/"
export PYTHONPATH="$PYTHONPATH:/raid/poggio/home/sanjanas/resnet-eccentricity/"

cd /raid/poggio/home/xboix/src/robustness-imagenet/official/resnet

singularity exec --nv /raid/poggio/home/xboix/containers/xboix-tensorflow.simg \
python imagenet_main.py  \
# --data_dir=/raid/poggio/home/xboix/data/imagenet-tfrecords \
--data_dir=/raid/poggio/home/sanjanas/data/resnet-ecc-data/numtrainex50_trial1 \
--num_gpus=8 \
--resnet_size=18 \
--batch_size=3072 \
# --factor=1 \
# --model_dir=/raid/poggio/home/xboix/src/robustness-imagenet/official/resnet/models/pawan/blur_bw_RF2blurL_bw_RF2color
--model_dir=/raid/poggio/home/sanjanas/data/resnet-ecc-data/models
