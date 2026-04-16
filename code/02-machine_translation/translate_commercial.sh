#!/bin/bash

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }


# define paths
DATAPATH=../../data/datasets/classifier_finetuning
CREDSPATH=./../../secrets # NOTE: need to place Google cloud and DeepL secrets in target folder


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# Düpont & Rachuj (2022) manifesto data 
DATAFILE=dupont_and_rachuj_2022_manifesto_sentences

message "Start translating Düpont & Rachuj (2022) dataset"
cp -n "${DATAPATH}/${DATAFILE}.tsv" "${DATAPATH}/${DATAFILE}_translated.tsv"
for service in google; do
	
	# set target language
	target_lang='en'
	creds_file="${CREDSPATH}/google.json"

	command="python3 translate.py \
		-i '${DATAPATH}/${DATAFILE}_translated.tsv' \
		--overwrite_output_file \
		--overwrite_target_column \
		--text_col 'text' \
		--lang_col 'lang' \
		--target_lang '$target_lang' \
		--translator '$service' \
		--api_key_file '$creds_file' \
		--batch_size 64 \
		--verbose "
	message "Translating text from '$DATAFILE' dataset with '$service'"
	eval $command
done


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# Lehmann & Zobel (2018) manifesto data 
DATAFILE=lehmann+zobel_2018_pimpo_positions

message "Start translating Lehmann and Zobel (2018) dataset"
cp -n "${DATAPATH}/${DATAFILE}.tsv" "${DATAPATH}/${DATAFILE}_translated.tsv"
for service in google deepl; do
	
	# set target language
	target_lang='en'
	[ "$service" == "deepl" ] && target_lang='EN-GB'

	# set credentials file
	creds_file="${CREDSPATH}/${service}.txt"
	[ "$service" == "google" ] && creds_file="${CREDSPATH}/google.json" # need to use API key JSON for Google API

	command="python3 translate.py \
		-i '${DATAPATH}/${DATAFILE}_translated.tsv' \
		--overwrite_output_file \
		--overwrite_target_column \
		--text_col 'text' \
		--lang_col 'lang' \
		--target_lang '$target_lang' \
		--translator '$service' \
		--api_key_file '$creds_file' \
		--batch_size 128 \
		--verbose "
	message "Translating text from '$DATAFILE' dataset with '$service'"
	eval $command
done


# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# Poljak (2022) parliamentary questions data 
DATAFILE=poljak_2022_attack_data
# note: need to use low batch size to avoid DeepL "413 Request Entity Too Large" API error

message "Start translating Poljak (2022) dataset"
cp -n "${DATAPATH}/${DATAFILE}.tsv" "${DATAPATH}/${DATAFILE}_translated.tsv"
for service in google deepl; do
	
	# set target language
	target_lang='en'
	[ "$service" == "deepl" ] && target_lang='EN-GB'

	# set credentials file
	creds_file="${CREDSPATH}/${service}.txt"
	[ "$service" == "google" ] && creds_file="${CREDSPATH}/google.json" # need to use API key JSON for Google API

	command="python3 translate.py \
		-i '${DATAPATH}/${DATAFILE}_translated.tsv' \
		--overwrite_output_file \
		--overwrite_target_column \
		--text_col 'text' \
		--lang_col 'lang' \
		--target_lang '$target_lang' \
		--translator '$service' \
		--api_key_file '$creds_file' \
		--batch_size 16 \
		--split_sentences \
		--verbose " 
	message "Translating text from '$DATAFILE' dataset with '$service'"
	eval $command
	cp "${DATAPATH}/${DATAFILE}_translated.tsv" "${DATAPATH}/xx_backup/${DATAFILE}_translated.tsv"
done

# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
# Theocharis et al. (2016) tweets data 
DATAFILE=theocharis_et_al_2016_labeled_tweets

message "Start translating Theocharis et al. (2016)"
cp -n "${DATAPATH}/${DATAFILE}.tsv" "${DATAPATH}/${DATAFILE}_translated.tsv"
for service in google deepl; do
	
	# set target language
	target_lang='en'
	[ "$service" == "deepl" ] && target_lang='EN-GB'

	# set credentials file
	creds_file="${CREDSPATH}/${service}.txt"
	[ "$service" == "google" ] && creds_file="${CREDSPATH}/google.json" # need to use API key JSON for Google API

	command="python3 translate.py \
		-i '${DATAPATH}/${DATAFILE}_translated.tsv' \
		--overwrite_output_file \
		--overwrite_target_column \
		--text_col 'text' \
		--lang_col 'lang' \
		--target_lang '$target_lang' \
		--translator '$service' \
		--api_key_file '$creds_file' \
		--batch_size 128 \
		--verbose "
	message "Translating text from '$DATAFILE' dataset with '$service'"
	eval $command
done


message "Done!"
