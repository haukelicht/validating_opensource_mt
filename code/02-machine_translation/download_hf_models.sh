#!/bin/bash
if sinfo >/dev/null 2>&1; then
	module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6  eth_proxy
	export TRANSFORMERS_CACHE=/cluster/work/lawecon/Work/hlicht/transformer_models
fi
source ./translation_venv/bin/activate


declare -A model_map
model_map[facebook/m2m100_1.2B]="7b36184180524c1a1bbfa37f120a608046250b98"
model_map[facebook/m2m100_418M]="62d980b8566a7c30e96918baf450d6a7218aadec"
model_map[Helsinki-NLP/opus-mt-bg-en]="8a944b4e892d77f7442095cb960d5c1a794766a5"
model_map[Helsinki-NLP/opus-mt-ca-en]="b16de52276f0c791f0e4e93ed152481581c2c715"
model_map[Helsinki-NLP/opus-mt-cs-en]="577d2053d249b476aefec9524a37ec945ccbd331"
model_map[Helsinki-NLP/opus-mt-da-en]="0960cff506758772cf23186b941305e0afeb43c5"
model_map[Helsinki-NLP/opus-mt-de-en]="1a922f3b32a8e809e17a47d4b32142d8105924e5"
model_map[Helsinki-NLP/opus-mt-en-de]="6183067f769a302e3861815543b9f312c71b0ca4"
model_map[Helsinki-NLP/opus-mt-es-en]="c96e2c5399ebfae4fc43d9669556b9afa74bb69d"
model_map[Helsinki-NLP/opus-mt-et-en]="8affd1fe86798b57bc7c63ee9d7ead91c02ba090"
model_map[Helsinki-NLP/opus-mt-fi-en]="c4d8fff0f8f00b637f0578666f35cdacef354778"
model_map[Helsinki-NLP/opus-mt-fr-en]="b4a9a384c2ec68a224bbd2ee3fd5df0c71ca5b1b"
model_map[Helsinki-NLP/opus-mt-gl-en]="f61d093dcbe0737c6d8955a35ef7eb476193bb9a"
model_map[Helsinki-NLP/opus-mt-hu-en]="cdf0ec884dc05cf435cef756dbf05a91c877f105"
model_map[Helsinki-NLP/opus-mt-it-en]="42556a0848fc726f4d27399f20b19ff6f01afe11"
model_map[Helsinki-NLP/opus-mt-lv-en]="580a8981c4e566c1ca84b176d286c0c42b82a7c6"
model_map[Helsinki-NLP/opus-mt-nl-en]="48af999f2c59b10c05ca6e008dcedc07677a9b15"
model_map[Helsinki-NLP/opus-mt-pl-en]="7f2bb874fdfb6139f9842b91a9b75c4a6c93401c"
model_map[Helsinki-NLP/opus-mt-ru-en]="fbd6dc73284f95536648512cc21d57f19191961a"
model_map[Helsinki-NLP/opus-mt-sk-en]="dd1de0d53e787147ac621680946dd55966049354"
model_map[Helsinki-NLP/opus-mt-sv-en]="202cf6240046bd8b6d08c207ee751ffd630d7ba8"
model_map[Helsinki-NLP/opus-mt-tr-en]="19c65427cc2af5f191337d4899e0348c4af25902"
model_map[Helsinki-NLP/opus-mt-uk-en]="1b9a2a0d1aa2a329d8476d2be5c7b0b27d5397b0"

for model in "${!model_map[@]}"; do
	revision=${model_map[$model]}
	huggingface-cli download "$model" --revision "$revision"
done
