#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=mpaso_soma_8km        #Set the job name to "JobExample1"
#SBATCH -C cpu
#SBATCH --qos=regular 
#SBATCH --time=24:00:00          
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=128
#SBATCH --account=m4304
#SBATCH --output=log_soma_8km.txt         #Send stdout/err to "Example1Out.[jobID768

srun -n 512 ./ocean_model -n namelist.ocean -s streams.ocean
