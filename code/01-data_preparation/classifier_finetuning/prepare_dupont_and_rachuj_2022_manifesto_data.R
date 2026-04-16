# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Get annotated natural sentences in machine-translated subset of 
#'          CMP corpus used by Düpont and Rachuj (2022)
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# load required packages
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(lubridate)
library(stringr)

input_path <- file.path("data", "exdata", "dupont_and_rachuj_2022")
output_path <- file.path("data", "datasets", "classifier_finetuning")

# download and read the data ----

#' @note: The original data comes from Düpont & Rachuj (2022).
#'        I take the cleaned data from the replication materials of Licht (2023)
#'         available through CodeOcean (https://codeocean.com/capsule/6171197/tree)
fp <- file.path(input_path, "dupont_and_rachuj_2022_coded_manifesto_sentences.rds")
if (!file.exists(fp)) {
  url <- "https://files.codeocean.com/files/verified/03843374-51b2-4ddf-b135-020a7b72471b_v1.0/data/input/dupont%2Brachuj/cleaned/coded_manifesto_sentences.tibble"
  download.file(url = url, destfile = fp)
}

# read
df <- read_rds(fp)

# inspect
glimpse(df)
# notes: 
#  - the English translations in 'text_en' come from Google Translate (obtained by Düpont & Rachuj in 2019)
#  - the English translations in 'text_en_m2m' come from the 418 million parameter M2m model
#  - we will drop the 'text_en_m2m' values for reproducibility's sake

count(df, domain_name_short, domain_code)
# note: Licht (2023) discarded the 'uncoded' samples
table(df$rile)
# note: Licht (2023) discarded the 'uncoded' samples

df |> 
  filter(rile != "uncoded") |> 
  count(domain_name_short, rile) |> 
  pivot_wider(names_from = rile, values_from = n)
# note: 'freedem' and 'econ' easy for issue-specific position/stance classification

# break down by country and language
overview <- df |> 
  group_by(language, country_iso3c) |> 
  summarise(
    n_manifestos = n_distinct(manifesto_id),
    n_sentences = n_distinct(sentence_id),
    n_characters = sum(nchar(trimws(text))),
    expected_translation_cost = (n_characters/1e6)*20
  )
overview
sum(overview$expected_translation_cost)

# fix known errors in text ----

replace <- c(
  '(?<!")"{2,}(?!")' = '"',
  "'{2}" = '"',
  "&#39;" = "'",
  "&quot;" = '"',
  "&amp;" = "&",
  "<U\\+FB01>" = "fi",
  "<U\\+FB02>" = "fl",
  "<U\\+201F>" = "'",
  "<U\\+2009>" = " ",
  "<U\\+2028>" = " ",
  "<U\\+FFFD>" = " ",
  "<U\\+[^>]+>" = "",
  "\\s+" = " "
)

df$text <- df$text |> 
  str_replace_all(replace) |> 
  str_trim()

df$text_en <- df$text_en |> 
  str_replace_all(replace) |>
  str_trim()

# extract relevant columns ----

with(df, table(domain_name_short == "uncoded", rile == "uncoded"))

out <- df |> 
  filter(domain_name_short != "uncoded") |> 
  select(
    country_iso3c, 
    sentence_id, 
    # text
    lang = lang_code, 
    text, 
    text_mt_google_old = text_en,
    # labels
    topic = domain_name_short,
    rile
  ) |> 
  mutate(
    econ_position = ifelse(topic != "econ", NA_character_, rile),
    freedem_position = ifelse(topic != "freedem", NA_character_, rile)
  )

table(out$lang)

# write to disk ----

fp <- file.path(output_path, "dupont_and_rachuj_2022_manifesto_sentences.tsv")
if (!file.exists(fp)) 
  write_tsv(out, fp)
