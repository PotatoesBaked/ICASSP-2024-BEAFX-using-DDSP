#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=16G

#SBATCH --output=./results/compressor/compressor.out
#SBATCH --error=./results/compressor/compressor.err

set -x

config_path=configs/compressor/compressor_Rf100.yaml
experience_name=compressor/compressor_Rf100_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name 

experience_name=compressor/compressor_Rf100_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --causal=False

config_path=configs/compressor/compressor_Rf300.yaml
experience_name=compressor/compressor_Rf300_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name 

experience_name=compressor/compressor_Rf300_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --causal=False

config_path=configs/compressor/compressor_Rf1000.yaml
experience_name=compressor/compressor_Rf1000_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name 

experience_name=compressor/compressor_Rf1000_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --causal=False

config_path=configs/compressor/compressor_Rf3000.yaml
experience_name=compressor/compressor_Rf3000_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name 

experience_name=compressor/compressor_Rf3000_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --causal=False
