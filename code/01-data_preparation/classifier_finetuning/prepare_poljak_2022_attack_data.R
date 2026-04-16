# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Prepare labeled data based on replication data from Poljak (2022)
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup -----

library(readxl)
library(readr)
library(dplyr)
library(purrr)
library(tidyr)
library(stringr)
library(cld2)
library(dataverse)
library(ggplot2)

input_path <- file.path("data", "exdata", "poljak_2022")
output_path <- file.path("data", "datasets", "classifier_finetuning")

# read the data ----

col_types <- c(
  "numeric",
  "text",
  "numeric",
  "text",
  "numeric",
  "date",
  "numeric",
  "numeric",
  "numeric",
  "date",
  "text",
  "logical",
  "text",
  "text",
  "numeric",
  "numeric",
  "logical",
  "numeric",
  "text",
  "logical",
  "logical",
  "numeric",
  "logical",
  "logical",
  "logical",
  rep("text", 24)
)

fp <- file.path(input_path, "attack_data.xlsx")
df <- read_xlsx(fp, col_types = col_types)

# fix text encodings ----

# guess encodings
encs <- df |> 
  transmute(
    id,
    speech,
    enc_unknown = map_chr(speech, Encoding) == "unknown",
    enc = stringi::stri_enc_detect(speech)
  )

# tabulate
table(encs$enc_unknown)
encs |> 
  filter(enc_unknown) |> 
  unnest(cols = enc) |> 
  group_by(id) |> 
  slice_head(n=1) |> 
  ungroup() |> 
  count(Encoding)

reencode_map <- encs |> 
  transmute(
    id,
    enc_unknown,
    enc = ifelse(
      enc_unknown,
      map_chr(enc, ~.$Encoding[[1]]),
      "UTF-8"
    )
  )
  
df <- df |> 
  left_join(reencode_map) |> 
  filter(!is.na(enc)) |> 
  mutate(
    speech_reencoded = case_when(
      !enc_unknown ~ speech,
      !is.na(enc) ~ map2_chr(speech, enc, ~iconv(.x, from = .y, to = "UTF-8")),
      TRUE ~ NA_character_
    )
  )
  
df |> 
  filter(enc_unknown, enc != "UTF-8") |> 
  filter(language != "English") |> 
  sample_n(20) |> 
  select(language, speech, enc, speech_reencoded) |> 
  arrange(language) |> 
  View()

# report descriptives ----

nrow(df)
# total number of attacks contained in text
df |> 
  count(attack_total) |> 
  mutate(prop = n/sum(n))

table(df$language)
with(df, table(language, attack_total)) |> prop.table(margin = 1) |> round(3)

df |> 
  filter(language == "English") |> 
  filter(attack_total %in% 1:3) |> 
  group_by(attack_total) |> 
  sample_n(3) |> 
  ungroup() |> 
  select(speech, attack_total, object1, object2, object3) |> 
  rowwise() |> 
  group_split() %>%
  purrr::map_chr(function(row) {
    sprintf(
      "%s [[%s]]", 
      row$speech[[1]], 
      paste(na.omit(select(row, starts_with("object"))[[1]]), collapse = "; ")
    )
  }) |> 
  paste(collapse = "\n\n-----------\n\n") |> 
  cat()

# cross tab occurrence of trait and policy attacks
df |> 
  filter(attack_total > 0) |>
  mutate(across(policy_attack:incivility, as.logical)) |> 
  with(table(policy_attack, trait_attack, useNA = "ifany"))

df <- df |> 
  mutate(across(policy_attack:incivility, as.logical)) |> 
  mutate(
    attack_type = case_when(
      is.na(policy_attack) & is.na(trait_attack) ~ NA_character_, 
      policy_attack & trait_attack ~ "P+T",
      policy_attack ~ "P",
      trait_attack ~  "T",
      TRUE ~ NA_character_
    ),
    attack_type_w_incivility = case_when(
      is.na(policy_attack) & is.na(trait_attack) ~ NA_character_, 
      policy_attack & trait_attack & incivility ~ "P/T-IC" ,
      policy_attack & trait_attack & !incivility ~  "P/T-C",
      policy_attack & incivility ~  "P-IC",
      policy_attack & !incivility ~ "P-C" ,
      trait_attack & incivility ~  "T-IC",
      trait_attack & !incivility ~ "T-C" ,
      TRUE ~ NA_character_
    )
  ) 

df |> 
  filter(attack_total > 0) |> 
  with(table(language, attack_type)) |> 
  prop.table(margin = 1) |> 
  round(3)

# validate languages ----

df$lang_guess <- detect_language(df$speech_reencoded)
with(df, table(lang_guess, language))

idxs <- which(with(df, lang_guess == "af"))
df[idxs, c("speech_reencoded")] # language correct

idxs <- which(with(df, lang_guess == "bs")) # Croation systematically recognized as Bosnian
df[idxs, c("speech_reencoded")] # language correct
df[idxs, "lang"] <- "bs"

idxs <- which(with(df, lang_guess == "en" & language != "English")) 
df[idxs, c("speech_reencoded")] # original language correct!

idxs <- which(with(df, lang_guess == "fr" & language != "French")) 
# View(df[idxs, c("speech_reencoded")]) # detected language correct!
df[idxs, "lang"] <- "fr"

idxs <- which(with(df, lang_guess == "nl" & language != "Dutch")) 
# View(df[idxs, c("speech_reencoded")]) # detected language correct!
df[idxs, "lang"] <- "nl"

df <- df |> 
  mutate(
    lang = case_when(
      !is.na(lang) ~ lang,
      language == "Croatian" ~ "hr",
      language == "Dutch" ~ "nl",
      language == "English" ~ "en",
      language == "French" ~ "fr",
    )
  )

table(df$lang)

# spot and remove "speeches" with multiple languages ----

table(duplicated(df$id))

sents <- df |> 
  transmute(id, lang, speech_reencoded) |> 
  mutate(sentence = speech_reencoded |> str_replace_all("\\s+", " ") |> stringi::stri_split_boundaries(type = "sentence")) |> 
  select(-speech_reencoded) |> 
  unnest(sentence) |> 
  mutate(lang_guess = detect_language(sentence))

top_prop <- function(x, na.rm = TRUE) {
  stopifnot(is.atomic(x))
  if (na.rm)
    x <- na.omit(x)
  x <- sort(table(x, useNA = "ifany")/length(x), decreasing = TRUE)[1]
  return(as.numeric(x))
}

topk_props <- function(x, k = 2, na.rm = TRUE) {
  stopifnot(is.atomic(x))
  if (na.rm)
    x <- na.omit(x)
  x <- sort(table(x, useNA = "ifany")/length(x), decreasing = TRUE)
  return(x[1:min(length(x), 2)])
}

entropy <- function(x, na.rm = TRUE) {
  stopifnot(is.atomic(x))
  if (na.rm)
    x <- na.omit(x)
  x <- table(x, useNA = "ifany")/length(x)
  -sum(x*log2(x))
}

tmp <- sents |> 
  group_by(id) |> 
  summarise(
    n_sents = n(),
    top_lang_guess_prop = top_prop(lang_guess),
    lang_guesses_entropy = entropy(lang_guess),
    top2_langs = list(topk_props(lang_guess))
  )

tmp
tmp$top2_langs[[1]]

table(lengths(tmp$top2_langs))

table(df$lang %in% c("fr", "nl"))

tmp_langs_contained <- tmp |> 
  filter(lengths(top2_langs) > 1) |> 
  # filter(top_lang_guess_prop < .75) |>
  mutate(
    lang = map(top2_langs, as.list),
    top2_langs = NULL
  ) |> 
  unnest_wider(lang, names_sep = "_") |> 
  left_join(select(df, id, lang)) #|>
  

tmp_langs_contained |> 
  group_by(lang) |> 
  summarise(across(matches("^lang_[a-z]{2}$"), sum, na.rm = TRUE)) |> 
  pivot_longer(-lang) |> 
  filter(value > 10) |> 
  arrange(lang, desc(value)) |> 
  filter(value > 1) |> 
  pivot_wider()

# "en" in supposedly "fr"?
cases <- tmp_langs_contained |> 
    filter(
      lang %in% c("fr"),
      lang_en > .1
    ) |> 
    arrange(n_sents, id) 
 
select(cases, id, lang, lang_en) |> left_join(sents) |> View()
# almost all correct ("fr")
# but ID 2102 contains English (not a problem for translation)
# # subset(df, id == 2102, speech)[[1]] |> cat()
remove_ids <- c(2102)

# "nl" in supposedly "fr"?
cases <- tmp_langs_contained |> 
    filter(
      lang %in% c("fr"),
      lang_nl > .1
    ) |> 
    arrange(n_sents, id) 
 
select(cases, id, lang, lang_nl) |> left_join(sents) |> View()

# these contain fr and nl
remove_ids <- c(remove_ids, 674, 1473, 1614, 2047, 2395, 2711, 3414, 3546, 3571, 3708, 5461, 5520, 5936, 6235)

# "en" in supposedly "nl"?
cases <- tmp_langs_contained |> 
  filter(
    lang %in% c("nl"),
    lang_en > .1
  ) |> 
  arrange(n_sents, id) 

select(cases, id, lang, lang_en) |> left_join(sents) |> View()
# all correct ("nl")

  
# "fr" in supposedly "nl"?
cases <- tmp_langs_contained |> 
  filter(
    lang %in% c("nl"),
    lang_fr > .1
  ) |> 
  arrange(n_sents, id) 

select(cases, id, lang, lang_fr) |> left_join(sents) |> View()
# these indeed all contain fr and nl

cases |> 
  distinct(id) |> 
  left_join(
    select(tmp_langs_contained, id, lang_fr, lang_nl)
  ) |> 
  pivot_longer(-id) |> 
  ggplot(aes(x = value, fill = name)) + 
    geom_histogram(alpha = .5)
# a lot of them are actually in nl:fr ratios in range 1:3 to 3:1

remove_ids <- c(remove_ids, unique(cases$id))
length(remove_ids)


# "af" (afrikans) in supposedly "nl"?
cases <- tmp_langs_contained |> 
  filter(
    lang %in% c("nl"),
    lang_af > .1
  ) |> 
  arrange(n_sents, id) 

select(cases, id, lang, lang_af) |> left_join(sents) |> View()
# all look fine


# remove to-be-removed speeches based on ID
df <- filter(df, !id %in% remove_ids)


# join translations ----

# the speech translations (Google Translate) are on dataverse https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0EPYVN
Sys.setenv("DATAVERSE_SERVER" = "dataverse.harvard.edu")
Sys.setenv("DATAVERSE_ID" = "ParlQuestionTime")
doi <- "doi:10.7910/DVN/0EPYVN"

table(df$con)["Croatia"]

## Croatia ----

croatia_translations <- get_dataframe_by_name(
  filename = "QuestionTimeSpeech v.1_Croatia.xlsx",
  dataset = doi,
  original = TRUE,
  .f = readxl::read_xlsx
)

croatia_speeches_translated <- df |> 
  filter(con == "Croatia") |> 
  select(id, qt_id, speech) |> 
  left_join(
    distinct(croatia_translations, speech, speech_english),
    # note: the translated data has no speech IDs, so need to join on speech text
    by = "speech"
  )

# any not matched?
table(is.na(croatia_speeches_translated$speech_english))
# great!

# check uniqueness
table(df$con)["Croatia"] == nrow(croatia_speeches_translated)

## Belgium ----

belgium_translations <- get_dataframe_by_name(
  filename = "QuestionTimeSpeech v.1_Belgium.xlsx",
  dataset = doi,
  original = TRUE,
  .f = readxl::read_xlsx
)

belgium_speeches_translated <- df |> 
  filter(con == "Belgium") |> 
  # keep language information to see if mistakes in language indicator in original data
  #  affect translation quality
  select(id, qt_id, speech, language, lang) |> 
  left_join(
    belgium_translations |> 
      distinct(speech, speech_english) |> 
      mutate(
        # note: this pattern at end of some speeches caused issues
        speech = str_replace(speech, " \\(…\\): \\(\\.{3}\\)$", "")
      ),
    # note: the translated data has no speech IDs, so need to join on speech text
    by = "speech"
  )

# any not matched?
table(is.na(belgium_speeches_translated$speech_english))
# 3 missing!

problem_cases <- belgium_speeches_translated |> 
  filter(is.na(speech_english)) |> 
  pull(speech)

# fix manually (using Google Translate interactively)
i = 1
problem_cases[i]
idx <- which(belgium_speeches_translated$speech == problem_cases[i])
belgium_speeches_translated$speech_english[idx] <- "Mr. President, Madam Minister, a brief response in three parts.\r\n \r\nFirst, if I heard correctly, the minister cited bpost's commercial action, but I did not hear it to say that she regretted the events as they unfolded. This would mean, Madam Minister, that if you do not get involved as the responsible minister, tomorrow the SNCB will sponsor the travel of Blood & Honor for its concerts or the Lotto team will invite Marine Le Pen to follow the Tour de France , to take just two examples.\r\n \r\n\r\n \r\n"

i = 2
problem_cases[i]
idx <- which(belgium_speeches_translated$speech == problem_cases[i])
belgium_speeches_translated$speech_english[idx] <- "I repeat: there will be no layoffs! Don't confuse the issues! Nor will there be any modification to the status of railway workers or questioning of the thirty-six hour regime. In this context, which clearly disturbs socialists, I would like to ask three questions. (Brouhaha)\r\n\r\n"

i = 3
problem_cases[i] # only non-letter characters
idx <- which(belgium_speeches_translated$speech == problem_cases[i])
belgium_speeches_translated$speech_english[idx] <- problem_cases[i]

#verify 
table(is.na(belgium_speeches_translated$speech_english))

# check uniqueness
table(df$con)["Belgium"] == nrow(belgium_speeches_translated)


# check translations for speeches with wrong language indicator
set.seed(1234)
belgium_speeches_translated |> 
  filter(language == "Dutch", lang == "fr") |> 
  sample_n(20) |> 
  View()

set.seed(1234)
belgium_speeches_translated |> 
  filter(language == "French", lang == "nl") |> 
  sample_n(20) |> 
  View()
# seems to be no problem (Zeljko likely just used Google's built-in language detection)

## add to data frame ----

df <- df |> 
  left_join(
    bind_rows(
      "Croatia" = select(croatia_speeches_translated, id, qt_id, speech_english),
      "Belgium" = select(belgium_speeches_translated, id, qt_id, speech_english),
      "UK" = df |> filter(con == "UK") |> select(id, qt_id, speech_english = speech_reencoded),
      .id = "con"
    )
  )

# verify
table(is.na(df$speech_english))

# export selected columns to TSV ----

glimpse(df)

# check if "objects" can be concatenated with comma
select(df, matches("object\\d+")) |> 
  unlist() |> 
  na.omit() |> 
  grepl(",", x = _) |> 
  table()

# concat targets
df <- df |> 
  rowwise() |> 
  mutate(
    targets = ifelse(
      attack_total == 0,
      NA_character_,
      paste(na.omit(c_across(matches("^object\\d+$"))), collapse = ",")
    )
  ) |> 
  ungroup() 


# inspect
df$targets |> head()
df |> 
  filter(attack_total > 1, language == "English") |> 
  select(attack_total, targets, speech) |> 
  sample_n(10) 

# # check text problems
# df$speech_reencoded |> str_extract_all("<U\\+[^>]>") |> unlist() |> table()
# df$speech_english |> str_extract_all("<U\\+[^>]>") |> unlist() |> table()
# df$speech_reencoded |> str_extract_all("'+") |> unlist() |> table()
# df$speech_english |> str_extract_all("'+") |> unlist() |> table()
# df$speech_reencoded |> str_extract_all('"+') |> unlist() |> table()
# df$speech_english |> str_extract_all('"+') |> unlist() |> table()
# df$speech_reencoded |> str_extract_all("&[^;];+?;") |> unlist() |> table()
# df$speech_english |> str_extract_all("&[^;];+?;") |> unlist() |> table()

clean_patterns <- c(
  # # Interruption indicator
  # "\\h+[Interruption]\\h+" = " ",
  # line break(s) to white space
  "(\\s*\\v+\\s*)+" = " ",
  # horizontal separator(s) to white space
  "\\h+" = " "
)

# test
test <- "&V, and so on, are saying to each other at this moment?\r\n\r\nWhat is this contempt for the world of work? It's unbearable!\r\n\r\nMr. Prime Minister, admit that in your answer you left us with little hope. We know that the workers are worried about the measures that you are preparing with Mr. Peeters: the 45-hour law; "
str_replace_all(test, clean_patterns)

out <- df |> 
  filter(language != "unknown") |> 
  # remove 106 "speeches" with only non-word characters
  filter(!grepl("^\\W*$", speech_reencoded)) |> 
  transmute(
    speech_id = id,
    country = con,
    country_iso3c = c("Belgium" = "BEL", "Croatia" = "HRV", "UK" = "GBR")[con],
    lang,
    text = speech_reencoded |> str_replace_all(clean_patterns) |> str_trim(),
    attack_binary, attack_total,
    policy_attack, trait_attack, attack_type, 
    incivility,
    targets,
    text_mt_google_old = speech_english |> str_replace_all(clean_patterns) |> str_trim()
  ) |> 
  mutate(
    # there are some sentence boundary problems in the Croatian texts that cause errors when translating with M2M
    #  so I fix them
    text = ifelse(
      lang == "hr",
      str_replace_all(text, c("(?<=\\p{Ll})([\\.?!,])(?=\\p{L}\\S+)" = "\\1 ")),
      text
    )
  )

fp <- file.path(output_path, "poljak_2022_attack_data.tsv")
if (!file.exists(fp))
  write_tsv(out, fp)

