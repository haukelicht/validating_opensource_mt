# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Get annotated sentences in machine-translated subset of CMP corpus
#' @author Hauke Licht
#' @note   You need to obtain a CMP API key to run this code
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup -----

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(manifestoR) # was 1.5.0, now 1.6.0
library(cld2)
library(ISOcodes)

## API key ----

secrets_path <- Sys.getenv("SECRETS_PATH", ".")
# NOTE: you need to obtain a CMP API key to run this code
key_file <- file.path(secrets_path, "manifesto_apikey.txt")
if (!file.exists(key_file) || file.info(key_file)$size == 0)
  stop("You need to obtain a CMP API key and store it in '<secrets_path>/manifesto_apikey.txt'.")
mp_setapikey(file.path(secrets_path, "manifesto_apikey.txt"))

## paths ----

data_path <- file.path("data", "datasets")

# get the raw data -----

# get docs that have English translations
doclist <- filter(mp_metadata(TRUE), translation_en)
# keep the docs that have annotations
these <- doclist[doclist$annotations, ]

# get the quasi-sentnece level data
docs <- mp_corpus(ids = these)
docs_en <- mp_corpus(ids = these, translation = "en")

# parse
docs_df <- tibble(value = as.list(docs)) |> 
  mutate(
    meta = value |> map("meta") |> map(compose(as.data.frame.list, as.list)),
    content = value |> map("content"),
    value = NULL
  ) |> 
  unnest_wider(meta) |>
  unnest(content) |> 
  group_by(manifesto_id) |>
  mutate(pos = row_number()) |>
  ungroup()

docs_en_df <- tibble(value = as.list(docs_en)) |> 
  mutate(
    manifesto_id = value |> map("meta") |> map_chr("manifesto_id"),
    content = value |> map("content"),
    value = NULL
  ) |> 
  unnest(content) |> 
  group_by(manifesto_id) |>
  mutate(pos = row_number()) |>
  ungroup() |> 
  select(manifesto_id, pos, text_en = text)

# join 
df <- inner_join(docs_df, docs_en_df, by = c("manifesto_id", "pos"))

n_distinct(df$manifesto_id)
n_distinct(df$language)

# add quasi-sentence ID (for reproducibility)
qs_id_fmt <- sprintf("%%s_%%0%dd", nchar(max(df$pos)))
df$qs_id <- with(df, sprintf(qs_id_fmt, manifesto_id, pos))

# filter to relevant documents ----

# discard unlabeled quasi-sentences
df_labelled <- filter(
    df,
    !is.na(cmp_code),
    cmp_code != "H"
  ) 

nrow(df_labelled)/nrow(df)

#' Simple utility function to identify natural sentence quasi-sentences
is_natural_sentence <- function(text) {
  grepl("^\\p{Lu}", text, perl = TRUE) & grepl("[.!?:]\\s*$", text, perl = TRUE)
}

df_labelled$nat_sentence <- with(df_labelled, is_natural_sentence(text) & is_natural_sentence(text_en))

# what's the share of quasi-sentences that are natural sentences?
df_labelled |> 
  with(table(language, nat_sentence)) |> 
  prop.table(1) |> 
  round(3)

# subset to natural sentences
df_labelled <- df_labelled[df_labelled$nat_sentence, ]


nrow(df_labelled)/nrow(df)

# apply label scheme ----

codebook <- mp_codebook("MPDS2024a")

domain_abbreviations <- c(
  "External Relations" = "exrel",
  "Freedom and Democracy" = "fredem",
  "Political System" = "polsys",
  "Economy" = "econ",
  "Welfare and Quality of Life" = "welqu",
  "Fabric of Society" = "fabso",
  "Social Groups" = "socgr"
)

codebook <- codebook |> 
  select(code, domain_code, domain_name) |> 
  mutate(
    domain = domain_abbreviations[domain_name],
    rile = case_when(
      code == "000" ~ NA_character_,
      code %in% rile_l() ~ "left",
      code %in% rile_r() ~ "right",
      TRUE ~ "none"
    )
  )

df_labelled <- df_labelled |> 
  mutate(
    topic = codebook$domain[match(cmp_code, codebook$code)],
    rile = codebook$rile[match(cmp_code, codebook$code)]
  ) |> 
  filter(!is.na(topic) & !is.na(rile))

nrow(df_labelled)/nrow(df)

with(df_labelled, table(language, domain)) |> prop.table(1) |> round(3)

with(df_labelled, table(language, rile)) |> prop.table(1) |> round(3)

# clean up further ----

## remove duplicated texts ----

table(duplicated(df_labelled$text))
df_labelled <- df_labelled[!duplicated(df_labelled$text), ]

table(duplicated(df_labelled$text_en))
df_labelled <- df_labelled[!duplicated(df_labelled$text_en), ]

## remove short texts ----

idxs <- nchar(df_labelled$text, type = "width") <= 10
table(df_labelled$text[idxs]) |> sort(decreasing = TRUE)

df_labelled <- df_labelled[!idxs, ]

## remove very long texts ----

nchars <- nchar(df_labelled$text, type = "width")
ggplot2::qplot(nchars) + ggplot2::scale_x_log10()

t_ <- quantile(nchars, 0.9999)
subset(df_labelled, nchars > t_) |> 
  pull(text_en) |> 
  strsplit("\\s+") |> 
  lengths() |> 
  summary()
# should not be too long for transformers

# NOTE: no action taken here

## remove missing texts ----

table(is.na(df_labelled$text))
table(is.na(df_labelled$text_en))
# none

# NOTE: no action taken here


## get 2-letter language codes ----

lang2code <- c(
  "swedish" = "sv",
  "norwegian" = "no", # OPUS-MT can't translate to 'en'
  "danish" = "da",
  "finnish" = "fi",
  "french" = "fr",
  "dutch" = "nl",
  "german" = "de",
  "italian" = "it",
  "spanish" = "es",
  "greek" = "el", # OPUS-MT can't translate to 'en'
  "portuguese" = "pt", # OPUS-MT can't translate to 'en'
  "turkish" = "tr",
  "bulgarian" = "bg",
  "czech" = "cs",
  "estonian" = "et",
  "hungarian" = "hu",
  "latvian" = "lv", 
  "lithuanian" = "lt", # OPUS-MT can't translate to 'en'
  "romanian" = "ro", # OPUS-MT can't translate to 'en' 
  "polish" = "pl",
  "russian" = "ru",
  "slovak" = "sk",
  "slovenian" = "sl", # OPUS-MT can't translate to 'en'
  "ukrainian" = "uk"
)

df_labelled$lang <- lang2code[df_labelled$language]

## validate source language indicator ----

lang_guess <- detect_language(df_labelled$text)

df_labelled$valid_lang_indicator <- with(df_labelled, !is.na(lang_guess) & lang == lang_guess)

df_labelled |> 
  with(table(lang, valid_lang_indicator)) |> 
  prop.table(1) |> 
  round(3)

# NOTE: discard where language is uncertain
table(df_labelled$valid_lang_indicator, useNA = "always")
df_labelled <- df_labelled[df_labelled$valid_lang_indicator, ]

## replace space characters -----

df_labelled$text <- gsub("\\s+", " ", df_labelled$text)
df_labelled$text_en <- gsub("\\s+", " ", df_labelled$text_en)

## subset to relevant columns ----

cols <- c(
  # metadata
  "manifesto_id", "party", "date", "lang",
  # text ID
  "qs_id", 
  # labels
  "topic", "rile",
  # text data
  "text", "text_en"
)
df_labelled <- df_labelled[cols]

# rename to indicat translation source
df_labelled <- rename(df_labelled, text_mt_deepl = text_en)


# save to disk ----

nrow(df_labelled)
head(df_labelled)

fp <- file.path(data_path, "classifier_finetuning", "cmp_translations.tsv")
if (!file.exists(fp))
  write_tsv(df_labelled, fp)

# sample subset to avoid high translation duration ---- 

sort(table(df_labelled$lang), decreasing = TRUE)

# discard Latvian ('lv') because few samples
df_labelled_sample <- df_labelled[df_labelled$lang != "lv", ]

n_ <- min(table(df_labelled_sample$lang))
set.seed(1234)
df_labelled_sample <- df_labelled_sample |> 
  group_by(lang) |> 
  sample_n(n_) |> 
  ungroup() 

n_ <- nrow(df_labelled_sample)
total_seconds <- n_/avg_sentences_per_second
total_seconds/60 # minutes
total_seconds/(60*60) # hours

fp <- file.path(data_path, "classifier_finetuning", "cmp_translations_sample.tsv")
if (!file.exists(fp))
  write_tsv(df_labelled_sample, fp)


df_labelled_sample <- rename(df_labelled_sample, topic = domain)

count(df_labelled_sample, topic)

out <- df_labelled_sample |> 
  mutate(econ_position = ifelse(topic == "econ", rile, NA_character_)) |> 
  select(
    manifesto_id, party, date, lang, 
    qs_id, 
    topic, rile, econ_position,
    starts_with("text")
  ) 

fp <- file.path(data_path, "classifier_finetuning",  "cmp_translations_sample_translated.tsv")
if (!file.exists(fp))
  write_tsv(out, fp)

