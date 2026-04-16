#!/bin/bash
#SBATCH --job-name=translate_theocharis_et_al_2016
#SBATCH -t 08:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32G
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

# load modules
message 'loading modules and virtual environment'
module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6  eth_proxy
source ./translation_venv/bin/activate

CACHE_PATH="/cluster/work/lawecon/Work/hlicht"
export EASYNMT_CACHE="${CACHE_PATH}/easynmt2"
export TRANSFORMERS_CACHE="${CACHE_PATH}/transformer_models"

# define paths
DATAPATH=../../data/datasets/classifier_finetuning

# Theocharis et al. (2016) tweets data 
DATAFILE=theocharis_et_al_2016_labeled_tweets

message "Start translating Theocharis et al. (2016)"
cp -n "${DATAPATH}/${DATAFILE}.tsv" "${DATAPATH}/${DATAFILE}_translated.tsv"
for model in m2m_100_418M m2m_100_1.2B opus-mt; do
	
	# set batch size
	batch_size=128
	[ "$model" == "m2m_100_1.2B" ] && batch_size=32

	python3 translate.py \
		-i "${DATAPATH}/${DATAFILE}_translated.tsv" \
		--overwrite_output_file \
		--overwrite_target_column \
		--text_col 'text' \
		--lang_col 'lang' \
		--target_lang 'en' \
		--translator 'easynmt' \
		--model_name $model \
		--batch_size $batch_size \
		--verbose
done

message "Done!"
