#!/bin/bash

module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6  eth_proxy
source ./transformer_finetuning_venv/bin/activate

export TRANSFORMERS_CACHE=/cluster/work/lawecon/Work/hlicht/transformer_models

declare -A model_map
model_map[roberta-base]="e2da8e2f811d1448a5b465c236feacd80ffbac7b" # Dec  5  2024
model_map[xlm-roberta-base]="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089" # Dec  5  2024
model_map[cardiffnlp/twitter-roberta-base]="cbb417e9647b51504caf68cbe1af6bbf56da06b7" # Dec 22  2023
model_map[cardiffnlp/twitter-xlm-roberta-base]="4c365f1490cb329b52150ad72f922ea467b5f4e6" # Nov 21  2023

for model in "${!model_map[@]}"; do
	revision=${model_map[$model]}
	echo "huggingface-cli download $model --revision $revision"
done

