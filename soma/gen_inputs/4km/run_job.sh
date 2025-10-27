#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=mpaso_soma_4km        #Set the job name to "JobExample1"
#SBATCH -C cpu
#SBATCH --qos=regular 
#SBATCH --time=48:00:00          
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=128
#SBATCH --account=m4304
#SBATCH --output=log_soma_4km.txt         #Send stdout/err to "Example1Out.[jobID768

srun -n 1408 ./ocean_model -n namelist.ocean -s streams.ocean
