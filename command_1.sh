#!/bin/sh

#SBATCH --job-name=parser_gpu16
#SBATCH --output=/scratch/%u/out_files/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/scratch_files/%x-%N-%j.err   # Error file
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=trahman2@gmu.edu     # Put your GMU email address here
#SBATCH --mem=32G    # Total memory needed per task (units: K,M,G,T)
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2    # Number of GPUs needed
##SBATCH --nodelist=NODE078  # If you want to run on a specific node
#SBATCH --nodes=1
#SBATCH --tasks=1


## Run your program or script
module purge
module load cuda/11.7.0 
module load gcc/9.3.0 
##source ~/torch-with-cuda/bin/activate
#python vae.py 
##./batch_download.sh -f ids.txt -o /scratch/trahman2/ext_structures 
# python -u main_qm9.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_1_alpha
# #python egnn_clean.py
# ./batch_download.sh -f pdb_ids_new.txt -p
python main.py

