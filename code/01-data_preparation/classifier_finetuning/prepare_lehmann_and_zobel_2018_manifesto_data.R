# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Prepare CMP immigration/integration/other annotated data
#' @author Hauke Licht
#' @source https://manifesto-project.wzb.eu/down/datasets/pimpo/create_PImPo_with_verbatim.r
#'         https://manifesto-project.wzb.eu/information/documents/pimpo
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #


# setup  ----

renv::activate()
library(readr)
library(dplyr)
library(manifestoR)
library(countrycode)
library(cld2)
library(ISOcodes)

input_path <- file.path("data", "exdata", "lehmann_and_zobel_2018")
output_path <- file.path("data", "datasets", "classifier_finetuning")

## API key ----

secrets_path <- Sys.getenv("SECRETS_PATH", ".")
# NOTE: you need to obtain a CMP API key to run this code
key_file <- file.path(secrets_path, "manifesto_apikey.txt")
if (!file.exists(key_file) || file.info(key_file)$size == 0)
  stop("You need to obtain a CMP API key and store it in '<secrets_path>/manifesto_apikey.txt'.")
mp_setapikey(file.path(secrets_path, "manifesto_apikey.txt"))

# fix version (to ensure compatibility with Lehmann/Zobel data)
mp_use_corpus_version(versionid = 20150708174629)

# load data ----

fp <- file.path(input_path, "PImPo_qsl_wo_verbatim.csv")

if (!file.exists(fp)) {
  url <- "https://manifesto-project.wzb.eu/information/documents/pimpohttps://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_qsl_wo_verbatim.csv"
  download.file(url = url, destfile = fp)
}

mans_coded <- read_csv(fp)

glimpse(mans_coded)
#' Codebook (see https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf)
#' 
#' @param pos_corpus Position of quasi-sentence in manifesto
#' @param gs_1r Gold standard item in 1st round of crowd coding
#' @param gs_answer_1r Gold standard coding ("0 if it was classified as not 
#'         related to immigration or integration and 1 if it was regarded as related to one of these")
#' @param gs_2r Gold standard item in 2nd round of crowd coding
#' @param gs_answer_2q Gold standard coding (1 if related to immigration, 2 if related to integration)
#' @param gs_answer_3q Gold standard coding "direction" in 2nd round of crowd coding
#' @param num_codings_1r Number of codings in in 1st round of crowd coding
#' @param selection aggregate coding in 1st round: 1 if immigration/integration related, 0 otherwise
#' @param certainty_selection inter-coder agreement for 'selection' item
#' @param topic aggregate topic coding in 2nd round: 1 if immigration, 2 if integration
#' @param certainty_topic inter-coder agreement for 'selection' item
#' @param direction aggregate direction coding in 2nd round: 1 if positive, 0 if neutral, -1 if skeptical
#' @param certainty_direction inter-coder agreement for 'selection' item
#' @param manually_coded quasi-sentence not coded by the crowd but by authors (to fill accidental missings)

# inspect

# gold lables for immigration/integration vs. others coding
with(mans_coded, table(gs_answer_1r, selection, useNA = "ifany"))

# gold lables for immigration vss integration coding
mans_coded %>% 
  filter(selection == 1) %>% 
  with(table(gs_answer_2q, topic, useNA = "ifany"))

# query CMP API for quasi-sentence text data ----

# get documents to be queried from CMP API
docs <- mans_coded %>% 
  select(party, date) %>% 
  distinct()

# get manifesto data
mans <- mp_metadata(docs) %>%
  filter(annotations) %>%
  mp_corpus() %>%
  as.data.frame(with.meta = TRUE) %>%
  filter(text != ".", !is.na(cmp_code))

# get manifesto data metadata
dataset <- mp_maindataset()

# join data with metadata
mans <- dataset %>% 
  transmute(
    country,
    countryname,
    manifesto_id = paste(party, date, sep = "_"),
    election_date = edate,
    partyname
  ) %>% 
  right_join(
    mutate(mans, manifesto_id = paste(party, date, sep = "_"))
  )

mans

# get country IDS
cmp_countries <- mans %>% 
  distinct(country, countryname) %>% 
  mutate(country_iso3c = countryname(countryname, "iso3c"))

mans <- right_join(cmp_countries, mans)

# inspect and fix languages ----

with(mans, table(language))
#' Notes: 
#'  - for some data, language is not known (could be done with cld2)
#'  - very few obs for french

tmp <- mans %>% 
  filter(language == "NA") %>% 
  transmute(
    country_iso3c,
    manifesto_id,
    lang = detect_language(text)
  )

# note: assign manifestos based on pluarlity language
lang_detected <- tmp %>% 
  count(country_iso3c, manifesto_id, lang) %>% 
  group_by(country_iso3c, manifesto_id) %>% 
  mutate(prop = n/sum(n)) %>% 
  slice_max(order_by = prop, n = 1) %>% 
  ungroup() %>% 
  filter(prop >= 0.9)

with(lang_detected, table(country_iso3c, lang))
# note: languages also align with country codes

a3code_to_lang <- with(ISOcodes::ISO_639_3, setNames(tolower(Name), Id))
lang_to_a3code <- setNames(names(a3code_to_lang), a3code_to_lang)
a2code_to_lang <- with(filter(ISOcodes::ISO_639_2, !is.na(Alpha_2)), setNames(tolower(Name), Alpha_2))
lang_to_a2code <- setNames(names(a2code_to_lang), a2code_to_lang)

ISOcodes::ISO_639_3$eng[grep("Spanish", ISOcodes::ISO_639_3$eng)]


# change language codes to names
lang_detected$lang <- a2code_to_lang[lang_detected$lang]

# add missing language info
mans <- mans %>% 
  left_join(select(lang_detected, manifesto_id, lang)) %>% 
  mutate(
    language = ifelse(language == "NA", lang, language),
    lang = NULL
  )

with(mans, table(language))

# add language information to dataset
mans <- mans %>% 
  transmute(
    country_iso3c,
    party = as.integer(party),
    election_date,
    manifesto_id,
    lang = lang_to_a3code[language],
    qs_nr = pos,
    qs_id = sprintf("%s_%05d", manifesto_id, pos),
    text,
    cmp_code
  ) %>% 
  as_tibble()

# check: no duplicates
mans$qs_id %>% duplicated() %>% table()

# remove duplicates from labeled dataset ----

tmp1 <- mans_coded %>% 
  filter(!is.na(pos_corpus)) %>% 
  group_by(country, date, party, pos_corpus) %>% 
  filter(n() == 1) %>% 
  ungroup()

tmp2 <- mans_coded %>% 
  filter(!is.na(pos_corpus)) %>% 
  group_by(country, date, party, pos_corpus) %>% 
  filter(n() > 1) 
nrow(tmp1); nrow(tmp2)

mans_coded_cleaned <- tmp2 %>% 
  # remove duplicates
  summarise(
    idx_ = ifelse(all(is.na(gs_2r)), 1L, which.max(gs_2r)),
    gs_1r = gs_1r[idx_],
    gs_answer_1r = gs_answer_1r[idx_],
    selection = selection[idx_],
    gs_2r = gs_2r[idx_],
    gs_answer_2q = gs_answer_2q[idx_],
    topic = topic[idx_],
    gs_answer_3q = gs_answer_3q[idx_],
    direction = direction[idx_]
  ) %>% 
  select(-idx_) %>% 
  ungroup() %>% 
  bind_rows(select(tmp1, -starts_with("certainty_"), -manually_coded, -rn)) %>% 
  arrange(country, date, party, pos_corpus)
  
mans_coded_cleaned_recoded <- mans_coded_cleaned %>%
  mutate(
    # immigration, integration, neither issue coding:
    issue = case_when(
      # - use round-1 gold standard coding if available
      !is.na(gs_1r) & gs_1r & gs_answer_1r == 0 ~ "other"
      # - if no valid label on either indicator, its neither,
      is.na(gs_2r) & is.na(topic) ~ "other"
      # # - else if valid gold standard label, use this,
      !is.na(gs_2r) & gs_2r & !is.na(gs_answer_2q) ~ ifelse(gs_answer_2q == 1, "immigration", "integration")
      # - else, use crowd-sourced label,
      !is.na(topic) ~ ifelse(topic == 1, "immigration", "integration"),
      TRUE ~ NA_character_
    ),
    issue = factor(issue, levels = c("immigration", "integration", "other"))
    # binary issue indicator,
    issue_binary = factor(ifelse(issue == "other", "other", "immigration/integration"), levels = c("immigration/integration", "other")),
    position = case_when(
      issue == "other" ~ NA_real_,
      !is.na(gs_answer_2q) ~ gs_answer_2q,
      !is.na(direction) ~ direction,
      TRUE ~ NA_real_
    )
    # note: there are 144 sentences with position = 2 (shouldn't be allowed): I recode them to 'supportive',
    position = factor(position, c(-1, 0, 1, 2), c("sceptical", "neutral", "supportive", "supportive"))
  ) %>% 
  select(country, date, party, pos_corpus, issue, issue_binary, position) 

# validate 
count(mans_coded_cleaned_recoded, issue, issue_binary)
count(mans_coded_cleaned_recoded, issue_binary, position)

# combine everything ----

out <- mans %>% 
  inner_join(
    mans_coded_cleaned_recoded %>% 
      mutate(
        manifesto_id = paste(party, date, sep = "_"),
        country = NULL,
        date = NULL
      ) ,
    by = c("manifesto_id", "party", "qs_nr" = "pos_corpus")
  )

# any duplicates?
out$qs_id %>% duplicated() %>% table()


# write to disk ----

fp <- file.path(output_path, "lehmann+zobel_2018_labeled_manifestos.tsv")
if (!file.exists(output_path))
  write_tsv(out, fp)
