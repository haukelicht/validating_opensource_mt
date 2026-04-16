# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Sample examples for analysis of translation discrepancy on 
#'          classifiers' classification accuracy  
#' @author Hauke Licht
#' 
#' @description: 
#'  1. We start with classifiers' test set predictions for the CMP Translations 
#'     corpus tasks. We take this corpus for several reasons.
#'      a) It covers more language families than the other datasets.
#'      b) It's commercial translations are likely of high quality because 
#'         - they were obtained relatively recently (very recent version of DeepL's closed-sourced MT system)
#'         - they were obtained with sentence context
#'      c) We have subset the corpus to intact natural sentences (i.e., 
#'         discarding much of the quasi-sentence spam in the CMP corpus) which 
#'         should make translations more reliable.
#'      d) Because we open-source translated texts without context, our comparison
#'         is conservative.
#'  2. We then take test set examples for which the DeepL-based classifiers' 
#'     predicted labels are correct given the the "true" labels (annotations).
#'     Given that the "true" labels have been assigned by coders who speak 
#'     texts' source language, we can assume that a correct classification of a 
#'     classifier fine-tuned on machine-translated texts is positively associated 
#'     with machine translation quality.
#'  3. Next, we get the open-source MT-based classifiers predicted labels for 
#'     these examples. The open-source MT-based classifiers predicted labels 
#'     might either agree with "true" labels (and hence, by design, the commercial 
#'     MT-based classifiers' predicted labels) or disagree. Because we focus on 
#'     cases where the commercial MT-based classifiers' predictions are correct, 
#'     we can more likely attribute open-source MT-based classifiers' error to 
#'     translation issues.
#'  We will use this data to compute the similarity of open-source model 
#'     translations to DeepL translations using BERTscore.
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

## load required packages ----

library(readr, quietly=TRUE, warn.conflicts=FALSE)
library(jsonlite, quietly=TRUE, warn.conflicts=FALSE)
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)
library(tidyr, quietly=TRUE, warn.conflicts=FALSE)
library(purrr, quietly=TRUE, warn.conflicts=FALSE)
library(stringr, quietly=TRUE, warn.conflicts=FALSE)

## paths ----

data_path <- file.path("data")
results_path <- file.path(data_path, "results")
utils_path <- file.path("code", "utils")

## mappings ----

source(file.path(utils_path, "mappings.R"))

mt_model_map_inv <- setNames(names(mt_model_map), mt_model_map)

# parse results files for test set classifications ----

# list the files that record classifiers' test set predictions
res_files <- list.files(
  file.path(results_path, "classifier_finetuning"), 
  pattern = "^cmp_translations_sample.+\\.json$", 
  full.names = TRUE
)

# helper function for parsin result (JSON) files 
parse_res_file <- function(fp, group.var.name = "lang") {
  
  dat <- read_json(fp)  
  
  preds <- dat$labels[!map_lgl(dat$labels, is.null)]
  preds <- as_tibble(map(preds, unlist))
  
  
  nm <- sub("\\.json$", "", basename(fp))
  nms <- str_split(nm, "-", n = 2)[[1]]
  
  out <- list(
    dataset = sub("^(.+?\\d{4})_.+", "\\1", nms[1], perl = TRUE),
    task = sub("^.+?\\d{4}_(.+)", "\\1", nms[1], perl = TRUE),
    mt_model = nms[2],
    predictions = preds
  )
  
  # account for file name differences
  if (grepl("^cmp_translations_sample", nm)) {
    out$dataset <- "cmp_translations_sample"
    out$task <- sub("cmp_translations_sample_([^-]+)-.+", "\\1", nm, perl = TRUE)
  }
  
  return(out)
}

# parse and combine into a data frame
res <- res_files |> 
  map(parse_res_file) |> 
  tibble(value = _) |> 
  unnest_wider(value) |> 
  filter(mt_model %in% mt_model_map_inv) |> 
  mutate(
    mt_model = factor(mt_model, names(mt_model_map), mt_model_map)
  ) 

# put DeepL-based and open-source MT-based classifiers' classification side by side -----

pairings <- map(unname(mt_model_map[4:6]), ~c(unname(mt_model_map["deepl"]), .))

# iterate over all pairs of MT models
preds <- map_dfr(pairings, function(pair) { #pair <- pairings[[2]]
  
  # subset predictions to relevant MT models 
  out <- res |> 
    filter(mt_model %in% pair) |> 
    filter(dataset %in% "cmp_translations_sample") |>
    group_by(dataset, task) |> 
    filter(n_distinct(mt_model) == 2)  |> 
    group_split() |> 
    map_dfr(function(dat) { # splits[[2]] -> dat
      
      a <- dat$predictions[dat$mt_model == pair[1]][[1]]
      b <- dat$predictions[dat$mt_model == pair[2]][[1]]
      
      if ("id" %in% names(a) & "id" %in% names(b)) {
        tmp <- left_join(
          a, b,
          by = c("label", "group", "id"),
          suffix = c("_commercial", "_other")
        )
      } else {
        tmp <- bind_cols(
          rename(a, pred_commercial = pred), 
          select(b, pred_other = pred),
        )
      }
      
      out <- tmp |> 
        mutate(language = language_iso2c_to_name[group]) |> 
        # # focus on case where commercial MT-based classifier is correct
        # filter() |> 
        mutate(
          other_model = pair[2],
          # create binary indicator if open-source/commercial MT-based classifier is correct
          commercial_is_correct = pred_commercial == label,
          opensource_is_correct = pred_other == label,
          lang = group,
          language = language_iso2c_to_name[lang]
        )
      
      return(bind_cols(distinct(dat[1:2]), out))
    }
    )
  
  return(out)
})

preds

fp <- file.path(data_path, "intermediate", "cmp_translations_sample_predicted_label_disagreements.tsv")
if (!file.exists(fp))
  write_tsv(preds, fp)

# NOTE: we have no OPUS-MT translations for CMP translations Corpus
count(preds, other_model, lang, language)

## identify unique translation model pairs in this subset ----

cases <- distinct(preds, other_model, qs_id = id, lang, language)
count(cases, other_model, language)

fp <- file.path(data_path, "datasets", "classifier_finetuning", "cmp_translations_sample_translated.tsv")
translations <- read_tsv(fp, show_col_types = FALSE)

cases <- cases |> 
  left_join(translations) |> 
  mutate(
    translation_a = text_mt_deepl,
    translation_b = ifelse(mt_model_map_inv[other_model] == "opus-mt", text_mt_opus_mt, text_mt_m2m_100_1.2b)
  ) |> 
  select(-starts_with("text_mt_"))

count(cases, other_model)

# export for BERTscore scroing and better translation classificaiton ----
fp <- file.path(data_path, "intermediate", "cmp_translations_sample_label_disagreement_cases.tsv")
if (!file.exists(fp))
  write_tsv(cases, fp)
