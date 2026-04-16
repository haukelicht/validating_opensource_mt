# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Clean codings from Theocharis et al. (2016)	  
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(purrr)
library(irr) # 0.84.1

packageVersion("irr")
input_path <- file.path("data", "exdata", "theocharis_et_al_2016")
output_path <- file.path("data", "datasets", "classifier_finetuning")

# helper functions

compute_kalpha <- function(x, item_col, coder_col, label_col, ...) {
  d <- select(x, item := {{item_col}}, coder := {{coder_col}}, lab := {{label_col}})
  d <- pivot_wider(d, names_from = "item", values_from = "lab")
  d <- as.matrix(d[-1])
  d <- d[rowSums(!is.na(d)) > 0, colSums(!is.na(d)) > 0]
  suppressWarnings(irr::kripp.alpha(d, ...))
}

tidy.irrlist <- function(x) {
  as_tibble(x[c("method", "subjects", "raters", "value")])
}

# read replication data ----

# define column types
col_types_ <- c(
  "X_unit_id" = "c",
  "X_created_at" = "?",
  "X_id" = "c",
  "X_started_at" = "?",
  "X_tainted" = "l",
  "X_channel" = "c",
  "X_trust" = "d",
  "X_worker_id" = "c",
  "X_country" = "c",
  "X_region" = "c",
  "X_city" = "c",
  "X_ip" = "c",
  "communication" = "f",
  "filter" = "f",
  "issue" = "f",
  "level" = "f",
  "polite" = "f",
  "political" = "f",
  "sentiment" = "f",
  "communication_gold" = "f",
  "filter_gold" = "f",
  "id" = "c",
  "issue_gold" = "f",
  "level_gold" = "f",
  "polite_gold" = "f",
  "political_gold" = "f",
  "screen_name" = "c",
  "sentiment_gold" = "f",
  "text" = "c",
  "country" = "c"
)

df <- read_csv(
  file.path(input_path, "tweet-codings.csv"),
  col_types = paste(col_types_, collapse = ""),
  quote = '"'
)

# clean the data ----

#' Clean columns with multilabel annotation
#' 
#' @param x the column containing the multilabel data
#' @param remove.labels names of label classes to be discarded 
#' @param recode.labels name-value pairs for recoding label classes
clean_multilabel <- Vectorize(function(x, remove.labels = NULL, recode.labels = NULL) {
  if (is.na(x)) 
    return(character())
  out <- stringr::str_split(x, pattern =  "\\s+")[[1]]
  out <- out[nchar(out) > 0]
  if (is.character(remove.labels))
    out <- out[!out %in% remove.labels]
  if (is.character(recode.labels) & !is.null(names(recode.labels)))
    for (i in seq_along(recode.labels))
      out[out == names(recode.labels[i])] <- recode.labels[i]
    out <- unique(out)
  return(out)
}, vectorize.args = "x", SIMPLIFY = FALSE)

df <- df |> 
  # convert to time stamps
  mutate(across(ends_with("_at"), mdy_hms)) |> 
  # rename figure8 meta data variables
  rename_all(~sub("^X_", "coding_", .)) |> 
  # drop "tainted" codings
  filter(!coding_tainted) |>
  mutate(
    # determine country ISO 3-character codes
    country_iso3c = case_when(
      country == "Germany" ~ "DEU",
      country == "Greece" ~ "GRC",
      country == "Spain" ~ "ESP",
      country == "UK" ~ "GBR",
      TRUE ~ NA_character_
    )
    # make 'political level' codings list of character vectors
    # , level_ = clean_multilabel(level, remove.labels = c("unclear", "none")),
    level_ = clean_multilabel(level)
    # make 'policy issues' codings list of character vectors,
    issue_ = clean_multilabel(issue)
  ) 

# inspect the data ----

## codings/annotations per tweet by country sample ----

df |> 
  group_by(country, coding_unit_id) |> 
  summarise(n_codings = n_distinct(coding_worker_id)) |> 
  count(country, n_codings) |> 
  pivot_wider(names_from = "n_codings", values_from = "n")

# inspect label columns ----

glimpse(df)
#' @note: Theocharis et al. have collected annotations of for tweets on multiple 
#'     dimensions in a conditional coding process.
#'     The coding dimensions are 
#'      - 'filter': categories to filter our spam tweets, "none" if non-spam tweet
#'      - if filter == "none"
#'        - 'communication': "broadcasting" or "engaging" communication? 
#'        - 'polite': "polite" or "impolite"
#'        - 'sentiment': "positive", "netural", "negative"
#'        - 'political': "political", "personal", "unclear"
#'        - if political == "political"
#'          - 'level' (multi-label): "eu", "national", and/or "subnational" or "unclear" or "none"
#'          - 'issue' (multi-label): 14 cats. (e.g., "campaign" or "economic")

## 'filter' ----

# the 'filter' indicator is there to filter out spam tweets 
table(df$filter, useNA = "ifany")
# note: only tweets with filter=="none" are annotated on other dimensions

tmp <- filter(df, filter == "none")

## 'communication' ----

ka <- tmp |> 
  group_by(coding_unit_id) |> 
  filter(n_distinct(coding_worker_id) > 1) |> 
  ungroup() |> 
  compute_kalpha(item_col = coding_unit_id, coder_col = coding_worker_id, label_col = communication) |> 
  tidy.irrlist()

ka

# communication is a binary indicators (in non-spam tweets) 
with(df, table(filter, communication, useNA = "ifany"))
with(tmp, round(prop.table(table(country_iso3c, communication), 1), 2))

## 'polite' ----

# 'polite' is a binary indicator (in non-spam tweets) 
with(df, table(filter, polite, useNA = "ifany"))
with(tmp, table(country_iso3c, polite))
# note: stringly imbalanced
with(tmp, round(prop.table(table(country_iso3c, polite), 1), 2))
# note: most negativity in GRC

## 'sentiment' ----

# 'sentiment' is an ordinal indicator (in non-spam tweets) 
with(df, table(filter, sentiment, useNA = "ifany"))
with(tmp, table(country_iso3c, sentiment))
with(tmp, round(prop.table(table(country_iso3c, sentiment), 1), 2))
# note: strong class imbalance
# note: could be made binary by discarding 'neutral' class

# note co-occurrence with 'polite'
with(tmp, table(polite, sentiment)) |> prop.table(1)
with(tmp, table(polite, sentiment)) |> prop.table(2)
with(tmp, chisq.test(polite == "impolite", sentiment == "negative"))
with(tmp, cor.test(as.integer(polite == "impolite"), as.integer(sentiment == "negative")))
# note: impolite tweets are very likely to have negative sentiment 
with(tmp, table(polite, sentiment, country_iso3c)) |> prop.table(c(1, 3))
# note: association weakest in Spain)  

## 'political' ----

# political is a categorical indicator (in non-spam tweets) 
with(df, table(filter, political, useNA = "ifany"))
with(tmp, round(prop.table(table(country_iso3c, political), 1), 2))
# note: strong label class imbalance (least pronounced in DEU)

tmp <-  filter(df, political == "political")

# detect tweets' languages ----

# patterns to remove before language detection
pats <- c(
  # URLS
  "(?<=\\s|^)https?://\\S+" = "", 
  # hashtags
  "(?<=\\s|^)#[A-Za-z0-9_]+" = "", 
  # mentions
  "(?<=\\s|^)[_\\.…]?@[A-Za-z0-9_]+" = ""
)

# helper function
get_language_score <- Vectorize(function(text, ...) {
  guess <- cld2::detect_language_mixed(text, ...)
  out <- guess$classification[1, c("code", "proportion")]
  if (!guess$reliabale) {
    out$code <- NA_character_
    out$proportion <- NA_real_
  }
  return(out)
}, vectorize.args = "text", SIMPLIFY = FALSE)

# detect languages
langs <- df |> 
  filter(filter == "none") |> 
  distinct(country_iso3c, coding_unit_id, text) |> 
  mutate(
    text_clean = str_replace_all(text, pats),
    lang = get_language_score(text_clean, plain_text = TRUE)
  ) |> 
  unnest(lang) |>
  mutate(
    expected_lang = case_when(
      country_iso3c == "DEU" & code == "de" ~ TRUE,
      country_iso3c == "ESP" & code == "es" ~ TRUE,
      country_iso3c == "GBR" & code == "en" ~ TRUE,
      country_iso3c == "GRC" & code == "el" ~ TRUE,
      TRUE ~ FALSE
    )
  )

with(langs, table(country_iso3c, expected_lang))
langs |> 
  filter(!expected_lang) |>
  with(table(code, country_iso3c))

valid_languages <- list(
  "DEU" = c("de", "en", "fr", "it", "es"), # checked cases with "no" and they are "de"
  "ESP" = c("es", "ca", "en", "gl", "pt"), # checked cases with "id", "rw" and they are "de"
  "GBR" = c("en"), 
  "GRC" = c("el", "en")
)

langs <- langs |> 
  mutate(
    valid_language = map2_lgl(code, country_iso3c, function(l, c) l %in% valid_languages[[c]]),
    # filter out cases where language detection failed
    valid_language = na_if(valid_language, is.na(code)),
    # filter out cases with uncertain language detection result
    valid_language = na_if(valid_language, proportion < .90)
  )
with(langs, table(valid_language, country_iso3c))

# clean tweet texts ----

# extract all characters used in tweets
chars <- df$text |> 
  tolower() |> 
  strsplit("") |> 
  unlist() |> 
  table() |> 
  sort(decreasing = TRUE)

# determine HTML entities
htlm_entities <- df$text |> 
  str_extract_all("\\b&[^;]+;") |> 
  unlist() |> 
  table() |> 
  sort(decreasing = TRUE)

html_entities_map <- c(
  "&amp;" = "&",
  "&gt;" = ">",
  "&lt;" = "<"
)

df <- df |> 
  mutate(
    # clean tweet text
    text_clean = str_replace_all(
      text,
      c(
        # remove URLs
        "https?://\\S+" = ""
        # remove superfluous white spaces ,
        "\\s+" = " "
        # standardize quote characters,
        "[“”ˮ„«»‘’'´ˊ΄]" = "'"
        # remove other special characters,
        "[\\\\—–—]" = ""
        # replace HTML entities,
        html_entities_map
      )
    )
  )

# aggregate at tweet level ----

## 'communication' ----

df_communication <- df |>
  filter(!is.na(communication) | filter_gold == "none") |> 
  group_by(coding_unit_id, country_iso3c) |> 
  summarise(
    # select most trusted sentiment coding (if multiple) 
    communication = communication[order(coding_trust, decreasing = TRUE)][1]
    # make gold labels unique,
    communication_gold = unique(communication_gold)
    # spam if,
    spam = case_when(
      # any gold label is not "none"
      all(filter_gold == "none") ~ FALSE
      # no "none" among codings,
      any("none" %in% filter) ~ FALSE,
      TRUE ~ TRUE
    ),
    .groups = "keep"
  ) |> 
  ungroup() |> 
  filter(!spam) |> 
  mutate(
    communication = if_else(is.na(communication_gold), communication, communication_gold),
    communication_gold = NULL,
    spam = NULL
  )

df_communication |> 
  with(table(country_iso3c, communication)) |> 
  prop.table(margin = 1) |> 
  round(3)

## 'polite' ----

df_polite <- df |>
  filter(!is.na(polite) | filter_gold == "none") |> 
  group_by(coding_unit_id, country_iso3c) |> 
  summarise(
    # select most trusted sentiment coding (if multiple) 
    polite = polite[order(coding_trust, decreasing = TRUE)][1]
    # make gold labels unique,
    polite_gold = unique(polite_gold)
    # spam if,
    spam = case_when(
      # any gold label is not "none"
      all(filter_gold == "none") ~ FALSE
      # no "none" among codings,
      any("none" %in% filter) ~ FALSE,
      TRUE ~ TRUE
    ),
    .groups = "keep"
  ) |> 
  ungroup() |> 
  filter(!spam) |> 
  mutate(
    polite = if_else(is.na(polite_gold), polite, polite_gold),
    polite = factor(polite == "polite", c(T, F), c("yes", "no")),
    polite_gold = NULL,
    spam = NULL
  )

df_polite |> 
  with(table(country_iso3c, polite)) |> 
  prop.table(margin = 1) |> 
  round(3)

## 'sentiment' ----

df_sentiment <- df |>
  filter(!is.na(sentiment) | filter_gold == "none") |> 
  group_by(coding_unit_id, country_iso3c) |> 
  summarise(
    # select most trusted sentiment coding (if multiple) 
    sentiment = sentiment[order(coding_trust, decreasing = TRUE)][1]
    # make gold labels unique,
    sentiment_gold = unique(sentiment_gold)
    # spam if,
    spam = case_when(
      # any gold label is not "none"
      all(filter_gold == "none") ~ FALSE
      # no "none" among codings,
      any("none" %in% filter) ~ FALSE,
      TRUE ~ TRUE
    ),
    .groups = "keep"
  ) |> 
  ungroup() |> 
  filter(!spam) |> 
  mutate(
    sentiment = if_else(is.na(sentiment_gold), sentiment, sentiment_gold)
    # , sentiment_binary = factor(na_if(sentiment, "neutral"), c("positive", "negative")),
    sentiment_binary = as.factor(case_when(sentiment == "positive" ~ "positive", sentiment == "negative" ~ "negative")),
    sentiment_gold = NULL,
    spam = NULL
  )

df_sentiment |> 
  with(table(country_iso3c, sentiment)) |> 
  prop.table(margin = 1) |> 
  round(3)

df_sentiment |> 
  with(table(country_iso3c, sentiment_binary)) |> 
  prop.table(margin = 1) |> 
  round(3)


## 'political' ----

df_political <- df |>
  filter(!is.na(political) | filter_gold == "none") |> 
  group_by(coding_unit_id, country_iso3c) |> 
  summarise(
    # select most trusted political coding (if multiple) 
    political = political[order(coding_trust, decreasing = TRUE)][1]
    # make gold labels unique,
    political_gold = unique(political_gold)
    # spam if,
    spam = case_when(
      # any gold label is not "none"
      all(filter_gold == "none") ~ FALSE
      # no "none" among codings,
      any("none" %in% filter) ~ FALSE,
      TRUE ~ TRUE
    ),
    .groups = "keep"
  ) |> 
  ungroup() |> 
  filter(!spam) |> 
  mutate(
    political = if_else(is.na(political_gold), political, political_gold),
    political_binary = factor(political == "political", c(T, F), c("yes", "no")),
    political_gold = NULL,
    spam = NULL
  )

df_political |> 
  with(table(country_iso3c, political)) |> 
  prop.table(margin = 1) |> 
  round(3)

df_political |> 
  with(table(country_iso3c, political_binary)) |> 
  prop.table(margin = 1) |> 
  round(3)


# combine ----

# discard "spam" tweets
out <- df |> 
  group_by(coding_unit_id, screen_name, country_iso3c) |> 
  summarise(
    filter_gold = unique(filter_gold)
    # spam if,
    spam = case_when(
      # any gold label is not "none"
      all(filter_gold == "none") ~ FALSE
      # no "none" among codings,
      any("none" %in% filter) ~ FALSE,
      TRUE ~ TRUE
    ),
    .groups = "keep"
  ) |> 
  ungroup() |> 
  # count(spam)
  filter(!spam) |> 
  select(coding_unit_id, screen_name, country_iso3c)

# join language information and subset to valid languages
out <- langs |> 
  filter(valid_language) |> 
  select(coding_unit_id, lang = code) |> 
  distinct() |> 
  inner_join(out) |> 
  select(country_iso3c, coding_unit_id, screen_name, lang)

# add cleaned tweet texts
out <- inner_join(out, distinct(df, coding_unit_id, text = text_clean))

# add labels
out <- out |> 
  left_join(df_communication) |> 
  left_join(df_polite) |> 
  left_join(df_sentiment) |> 
  left_join(df_political) |> 
  rename(focus = political, political = political_binary)

# write to disk ----

fp <- file.path(output_path, "theocharis_et_al_2016_labeled_tweets.tsv")
if (!file.exists(fp))
  write_tsv(out, fp)
