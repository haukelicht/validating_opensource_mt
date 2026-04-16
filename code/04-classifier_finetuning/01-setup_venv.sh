#!/bin/bash
if sinfo >/dev/null 2>&1; then
    module load eth_proxy python_cuda/3.11.6
    python -m venv --site-packages transformer_finetuning_venv
else
    python -m venv transformer_finetuning_venv
fi

source ./transformer_finetuning_venv/bin/activate
pip install --no-cache-dir -r requirements.txt
deactivate
