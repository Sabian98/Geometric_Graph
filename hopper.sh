#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=1-12:30:00 ## set for 1 day, 12 hours and 30 minutes
#SBATCH --output=/scratch/%u/out_files/%x-%N-%j.out
#SBATCH --error=/scratch/%u/scratch_files/%x-%N-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH   --cpus-per-task=48            # Request n   cores per node
 


#SBATCH --mem=10G
#SBATCH --time=1-23:59:00

#Load needed modules
# module purge
# module load cuda/10.2
# module load gcc/8.4.0

#Execute
# python generator.py
python main.py
