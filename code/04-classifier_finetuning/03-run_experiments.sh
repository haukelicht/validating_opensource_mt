#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --job-name=transformer_finetuning_experiments
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

# load modules
message 'Loading modules and virtual environment'
module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6  eth_proxy
source ./transformer_finetuning_venv/bin/activate

export TRANSFORMERS_CACHE=/cluster/work/lawecon/Work/hlicht/transformer_models

DATAPATH=../../data/datasets/classifier_finetuning
RESULTSPATH=../../data/results/classifier_finetuning

# we declare a mapping of text column names to model names
#  so that we use the multilingual model when we use column 
#  'text' and the English model otherwise
declare -A model_map
model_map[text]='xlm-roberta-base'
model_map[text_mt_deepl]='roberta-base'
model_map[text_mt_google]='roberta-base'
model_map[text_mt_google_old]='roberta-base'
# note: no 'text_mt_google_old' texts for Lehmann & Zobel data
model_map[text_mt_m2m_100_418m]='roberta-base'
model_map[text_mt_m2m_100_1.2b]='roberta-base'
model_map[text_mt_opus-mt]='roberta-base'


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# with Düpont & Rachuj (2022) manifesto data 
DATAFILE=dupont_and_rachuj_2022_manifesto_sentences_translated.tsv

# ... using 'topic' indicator
EXNAME=dupont_and_rachuj_2022_topic
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'sentence_id' \
		--text_col '$col' \
		--label_col 'topic' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,fin,fra,ita,nld,spa,swe' \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--epochs 2 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'rile' (position) indicator
EXNAME=dupont_and_rachuj_2022_rile
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'sentence_id' \
		--text_col '$col' \
		--label_col 'rile' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,fin,fra,ita,nld,spa,swe' \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--epochs 3 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'rile' (position) indicator
EXNAME=dupont_and_rachuj_2022_rile_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'sentence_id' \
		--text_col '$col' \
		--label_col 'rile' \
		--label_values 'left,right' \
		--pos_label 'left' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,fin,fra,ita,nld,spa,swe' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--epochs 3 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using economy 'rile' (position) indicator
EXNAME=dupont_and_rachuj_2022_econ_position_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'sentence_id' \
		--text_col '$col' \
		--label_col 'econ_position' \
		--label_values 'left,right' \
		--pos_label 'left' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,fin,fra,ita,nld,spa,swe' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using freedem 'rile' (position) indicator
EXNAME=dupont_and_rachuj_2022_freedem_position_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'sentence_id' \
		--text_col '$col' \
		--label_col 'freedem_position' \
		--label_values 'left,right' \
		--pos_label 'left' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,fin,fra,ita,nld,spa,swe' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done



# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# with Lehmann & Zobel (2018) manifesto data 
DATAFILE=lehmann+zobel_2018_pimpo_positions_translated.tsv

# ... using 'issue' indicator
EXNAME=lehmann+zobel_2018_pimpo_issue
for col in "${!model_map[@]}"; do
	
	[ "$col" == "text_mt_google_old" ] && continue

	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'qs_id' \
		--text_col '$col' \
		--label_col 'issue' \
		--pos_label 'immigration' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,eng,fin,fra,nld,spa,swe' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'position' indicator
EXNAME=lehmann+zobel_2018_pimpo_position
for col in "${!model_map[@]}"; do
	
	[ "$col" == "text_mt_google_old" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--text_col 'qs_id' \
		--text_col '$col' \
		--label_col 'position' \
		--label_values 'supportive,sceptical' \
		--pos_label 'supportive' \
		--filter_by_col 'lang' \
		--filter_value 'dan,deu,eng,fin,fra,nld,spa,swe' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done



# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# with Poljak (2022) parliamentary questions data 
DATAFILE=poljak_2022_attack_data_translated.tsv

# NOTE: first all without texts from Croatia (DeepL and OPUS-MT cannot translate hr->en and bs->en)

# ... using binary 'attack' indicator
EXNAME=poljak_2022_attack_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'speech_id' \
		--text_col '$col' \
		--label_col 'attack_binary' \
		--filter_by_col 'lang' \
		--filter_value 'en,nl,fr' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'attack type' indicator
EXNAME=poljak_2022_attack_type
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'speech_id' \
		--text_col '$col' \
		--label_col 'attack_type' \
		--filter_by_col 'lang' \
		--filter_value 'en,nl,fr' \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--epochs 8 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'incivility' indicator
EXNAME=poljak_2022_incivility
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'speech_id' \
		--text_col '$col' \
		--label_col 'incivility' \
		--filter_by_col 'lang' \
		--filter_value 'en,nl,fr' \
		--downsample_train_data --downsample_minority_ratio 0.333 --minority_label 'True' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--epochs 8 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done



# NOTE: now with 'hr' and 'bs' speeches from Croatia but without DeepL and OPUS-MT

# ... using binary 'attack' indicator
EXNAME=poljak_2022_attack_binary_w_croatia
for col in "${!model_map[@]}"; do
	
	# skip because no translation for hr->en and bs->en
	[ "$col" == "text_mt_deepl" ] && continue
	[ "$col" == "text_mt_opus-mt" ] && continue

	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'speech_id' \
		--text_col '$col' \
		--label_col 'attack_binary' \
		--filter_by_col 'lang' \
		--filter_value 'en,nl,fr,hr,bs' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'attack type' indicator
EXNAME=poljak_2022_attack_type_w_croatia
for col in "${!model_map[@]}"; do
	
	# skip because no translation for hr->en and bs->en
	[ "$col" == "text_mt_deepl" ] && continue
	[ "$col" == "text_mt_opus-mt" ] && continue

	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'speech_id' \
		--text_col '$col' \
		--label_col 'attack_type' \
		--filter_by_col 'lang' \
		--filter_value 'en,nl,fr,hr,bs' \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--epochs 8 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'incivility' indicator
EXNAME=poljak_2022_incivility_w_croatia
for col in "${!model_map[@]}"; do

	# skip because no translation for hr->en and bs->en
	[ "$col" == "text_mt_deepl" ] && continue
	[ "$col" == "text_mt_opus-mt" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'speech_id' \
		--text_col '$col' \
		--label_col 'incivility' \
		--filter_by_col 'lang' \
		--filter_value 'en,nl,fr,hr,bs' \
		--downsample_train_data --downsample_minority_ratio 0.333 --minority_label 'True' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--epochs 8 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# TWITTER DATA 
# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

declare -A model_map
model_map[text]='cardiffnlp/twitter-xlm-roberta-base'
model_map[text_mt_deepl]='cardiffnlp/twitter-roberta-base'
model_map[text_mt_google]='cardiffnlp/twitter-roberta-base'
# no 'text_mt_google_old' texts for Theocharis et al. (2016) data
model_map[text_mt_m2m_100_418m]='cardiffnlp/twitter-roberta-base'
model_map[text_mt_m2m_100_1.2b]=cardiffnlp/'twitter-roberta-base'
model_map[text_mt_opus-mt]=cardiffnlp/'twitter-roberta-base'


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# with Theocharis et al. (2016) EUP candidate tweets data 
DATAFILE=theocharis_et_al_2016_labeled_tweets_translated.tsv


# NOTE: first all without greek (because OPUT cannot translate el->en)

# ... using 'sentiment_binary' indicator
EXNAME=theocharis_et_al_2016_sentiment_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'sentiment_binary' \
		--pos_label 'broadcasting' \
		--text_preprocessing 'twitter' \
		--filter_by_col 'lang' \
		--filter_value 'de,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'sentiment' indicator
EXNAME=theocharis_et_al_2016_sentiment
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'sentiment' \
		--text_preprocessing 'twitter' \
		--filter_by_col 'lang' \
		--filter_value 'de,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'communication' indicator
EXNAME=theocharis_et_al_2016_communication
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'communication' \
		--pos_label 'broadcasting' \
		--text_preprocessing 'twitter' \
		--filter_by_col 'lang' \
		--filter_value 'de,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'polite' indicator
EXNAME=theocharis_et_al_2016_polite
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'polite' \
		--pos_label 'yes' \
		--text_preprocessing 'twitter' \
		--downsample_train_data --downsample_minority_ratio 0.5 --minority_label 'no' \
		--filter_by_col 'lang' \
		--filter_value 'de,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'political' indicator
EXNAME=theocharis_et_al_2016_political
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'political' \
		--pos_label 'yes' \
		--text_preprocessing 'twitter' \
		--downsample_train_data --downsample_minority_ratio 0.5 --minority_label 'no' \
		--filter_by_col 'lang' \
		--filter_value 'de,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# NOTE: now with greek but without OPUS-MT translations

# ... using 'sentiment_binary' indicator
EXNAME=theocharis_et_al_2016_sentiment_binary_w_greek
for col in "${!model_map[@]}"; do
	
	# skip because no translation for el->en
	[ "$col" == "text_mt_opus-mt" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'sentiment_binary' \
		--pos_label 'broadcasting' \
		--text_preprocessing 'twitter' \
		--filter_by_col 'lang' \
		--filter_value 'de,el,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'sentiment' indicator
EXNAME=theocharis_et_al_2016_sentiment_w_greek
for col in "${!model_map[@]}"; do
	
	# skip because no translation for el->en
	[ "$col" == "text_mt_opus-mt" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'sentiment' \
		--text_preprocessing 'twitter' \
		--filter_by_col 'lang' \
		--filter_value 'de,el,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'communication' indicator
EXNAME=theocharis_et_al_2016_communication_w_greek
for col in "${!model_map[@]}"; do
	
	# skip because no translation for el->en
	[ "$col" == "text_mt_opus-mt" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'communication' \
		--pos_label 'broadcasting' \
		--text_preprocessing 'twitter' \
		--filter_by_col 'lang' \
		--filter_value 'de,el,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'polite' indicator
EXNAME=theocharis_et_al_2016_polite_w_greek
for col in "${!model_map[@]}"; do
	
	# skip because no translation for el->en
	[ "$col" == "text_mt_opus-mt" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'polite' \
		--pos_label 'yes' \
		--text_preprocessing 'twitter' \
		--downsample_train_data --downsample_minority_ratio 0.5 --minority_label 'no' \
		--filter_by_col 'lang' \
		--filter_value 'de,el,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'political' indicator
EXNAME=theocharis_et_al_2016_political_w_greek
for col in "${!model_map[@]}"; do
	
	# skip because no translation for el->en
	[ "$col" == "text_mt_opus-mt" ] && continue
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'coding_unit_id' \
		--text_col '$col' \
		--label_col 'political' \
		--pos_label 'yes' \
		--text_preprocessing 'twitter' \
		--downsample_train_data --downsample_minority_ratio 0.5 --minority_label 'no' \
		--filter_by_col 'lang' \
		--filter_value 'de,el,es,en' \
		--training_batch_size 32 \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done




# we declare a mapping of text column names to model names
#  so that we use the multilingual model when we use column 
#  'text' and the English model otherwise
declare -A model_map
model_map[text]='xlm-roberta-base'
model_map[text_mt_deepl]='roberta-base'
model_map[text_mt_m2m_100_418m]='roberta-base'
model_map[text_mt_m2m_100_1.2b]='roberta-base'

# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# with Lehmann et al. (2024) CMP translations coprus
DATAFILE=cmp_translations_sample_translated.tsv

# ... using 'topic' indicator
EXNAME=cmp_translations_sample_topic
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'qs_id' \
		--text_col '$col' \
		--label_col 'topic' \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--epochs 2 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using 'rile' (position) indicator
EXNAME=cmp_translations_sample_rile
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'qs_id' \
		--text_col '$col' \
		--label_col 'rile' \
		--eval_metric 'f1_macro' \
		--eval_by 'lang' \
		--epochs 2 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using binary 'rile' (position) indicator
EXNAME=cmp_translations_sample_rile_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'qs_id' \
		--text_col '$col' \
		--label_col 'rile' \
		--label_values 'left,right' \
		--pos_label 'left' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--epochs 3 \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done


# ... using economy 'rile' (position) indicator
EXNAME=cmp_translations_sample_econ_position_binary
for col in "${!model_map[@]}"; do
	
	# get model name
	model=${model_map[$col]}
	
	# set indicator for what text source was used
	tsrc=${col#text_mt_}
	[ "$tsrc" == "text" ] && tsrc='multilingual' 
	
	# set command
	command="python3 finetune_classifier.py \
		--experiment_name '$EXNAME-$tsrc' \
		--data_file '$DATAPATH/$DATAFILE' \
		--out_file '$RESULTSPATH/$EXNAME-$tsrc.json' --out_file_overwrite \
		--model_name '$model' \
		--id_col 'qs_id' \
		--text_col '$col' \
		--label_col 'econ_position' \
		--label_values 'left,right' \
		--pos_label 'left' \
		--eval_metric 'f1' \
		--eval_by 'lang' \
		--training_batch_size 16 \
		--gradient_accumulation 2 \
		--print_results "
	message "Running experiment '$EXNAME' with text in column '$col' (using model '$model')"
	eval $command
done



message "Done!"
