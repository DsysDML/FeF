#!/bin/bash

# Ensure the script exits on error
set -e

# Validate the number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_data> <path_output>"
    exit 1
fi

# Assign positional arguments to variables
path_data="$1"
path_output="$2"

# Training commands
#fef train -d "$path_data/MNIST/MNIST_train.dat" -a "$path_data/MNIST/MNIST_train_ann.csv" -o "$path_output/MNIST" -l fef -H 1000 --nchains 5000 --gibbs_steps 10 --nepochs 100000
#fef train -d "$path_data/HGD/HGD_train.dat" -a "$path_data/HGD/HGD_train_ann.csv" -o "$path_output/HGD" -l fef -H 1000 --nchains 4507 --gibbs_steps 10 --nepochs 100000
#fef train -d "$path_data/CMPC/CMPC_train.dat" -a "$path_data/CMPC/CMPC_train_ann.csv" -o "$path_output/CMPC" -l fef -H 1000 --nchains 5000 --gibbs_steps 10 --nepochs 100000
#fef train -d "$path_data/SAM/SAM_train.fasta" -a "$path_data/SAM/SAM_train_ann.csv" -o "$path_output/SAM" -l fef -H 1000 --nchains 5000 --gibbs_steps 10 --nepochs 100000 --alphabet rna
#fef train -d "$path_data/GH30/GH30_train.fasta" -a "$path_data/GH30/GH30_train_ann.csv" -o "$path_output/GH30" -l fef -H 1000 --nchains 5000 --gibbs_steps 10 --nepochs 100000 --alphabet protein

#annadca train -d "$path_data/MNIST/MNIST_train.dat" -a "$path_data/MNIST/MNIST_train_ann.csv" -o "$path_output/MNIST" -l pcd -H 1000 --nchains 5000 --gibbs_steps 100 --nepochs 100000
#annadca train -d "$path_data/HGD/HGD_train.dat" -a "$path_data/HGD/HGD_train_ann.csv" -o "$path_output/HGD" -l pcd -H 1000 --nchains 4507 --gibbs_steps 100 --nepochs 100000
#annadca train -d "$path_data/CMPC/CMPC_train.dat" -a "$path_data/CMPC/CMPC_train_ann.csv" -o "$path_output/CMPC" -l pcd -H 1000 --nchains 5000 --gibbs_steps 100 --nepochs 100000
#annadca train -d "$path_data/SAM/SAM_train.fasta" -a "$path_data/SAM/SAM_train_ann.csv" -o "$path_output/SAM" -l pcd -H 1000 --nchains 5000 --gibbs_steps 100 --nepochs 100000 --alphabet rna
#annadca train -d "$path_data/GH30/GH30_train.fasta" -a "$path_data/GH30/GH30_train_ann.csv" -o "$path_output/GH30" -l pcd -H 1000 --nchains 5000 --gibbs_steps 100 --nepochs 100000 --alphabet protein

# Scoring commands (FEF)
fef score -d "$path_data/MNIST/MNIST_test.dat" -a "$path_data/MNIST/MNIST_test_ann.csv" -o "$path_output/MNIST/scoring" -l fef -p "$path_output/MNIST/fef_params.h5" -t 100 --num_records 20 --checkpoints "1000,10000,30000,60000,100000"
fef score -d "$path_data/HGD/HGD_test.dat" -a "$path_data/HGD/HGD_test_ann.csv" -o "$path_output/HGD/scoring" -l fef -p "$path_output/HGD/fef_params.h5" -t 100 --num_records 20 --checkpoints "1000,10000,30000,60000,100000"
fef score -d "$path_data/CMPC/CMPC_test.dat" -a "$path_data/CMPC/CMPC_test_ann.csv" -o "$path_output/CMPC/scoring" -l fef -p "$path_output/CMPC/fef_params.h5" -t 100 --num_records 20 --checkpoints "1000,10000,30000,60000,100000"
fef score -d "$path_data/SAM/SAM_test.fasta" -a "$path_data/SAM/SAM_test_ann.csv" -o "$path_output/SAM/scoring" -l fef -p "$path_output/SAM/fef_params.h5" -t 100 --num_records 20 --checkpoints "1000,10000,30000,60000,100000" --alphabet rna
fef score -d "$path_data/GH30/GH30_test.fasta" -a "$path_data/GH30/GH30_test_ann.csv" -o "$path_output/GH30/scoring" -l fef -p "$path_output/GH30/fef_params.h5" -t 100 --num_records 20 --checkpoints "1000,10000,30000,60000,100000" --alphabet protein

# Scoring commands (PCD)
#fef score -d "$path_data/MNIST/MNIST_test.dat" -a "$path_data/MNIST/MNIST_test_ann.csv" -o "$path_output/MNIST/scoring" -l pcd -p "$path_output/MNIST/pcd_params.h5" -t 100000 --num_records 20 --checkpoints "1000,10000,30000,60000,100000"
#fef score -d "$path_data/HGD/HGD_test.dat" -a "$path_data/HGD/HGD_test_ann.csv" -o "$path_output/HGD/scoring" -l pcd -p "$path_output/HGD/pcd_params.h5" -t 100000 --num_records 20 --checkpoints "1000,10000,30000,60000,100000"
#fef score -d "$path_data/CMPC/CMPC_test.dat" -a "$path_data/CMPC/CMPC_test_ann.csv" -o "$path_output/CMPC/scoring" -l pcd -p "$path_output/CMPC/pcd_params.h5" -t 100000 --num_records 20 --checkpoints "1000,10000,30000,60000,100000"
#fef score -d "$path_data/SAM/SAM_test.fasta" -a "$path_data/SAM/SAM_test_ann.csv" -o "$path_output/SAM/scoring" -l pcd -p "$path_output/SAM/pcd_params.h5" -t 100000 --num_records 20 --checkpoints "1000,10000,30000,60000,100000" --alphabet rna
#fef score -d "$path_data/GH30/GH30_test.fasta" -a "$path_data/GH30/GH30_test_ann.csv" -o "$path_output/GH30/scoring" -l pcd -p "$path_output/GH30/pcd_params.h5" -t 100000 --num_records 20 --checkpoints "1000,10000,30000,60000,100000" --alphabet protein
