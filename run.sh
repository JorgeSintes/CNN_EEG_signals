#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpuk80
### #BSUB -q gpuv100
### -- set the job Name --
#BSUB -J CNN_paper
### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=shared"
### #BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

### request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"

### -- set the email address --
### please uncomment the following line and put in your e-mail address,
### if you want to receive e-mail notifications on a non-default address
#BSUB -u jorge.sintes@gmail.com

### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o output_%J.out
### #BSUB -e gpu_%J.err
# -- end of LSF options --

# here follow the commands you want to execute
python cross_validation.py
