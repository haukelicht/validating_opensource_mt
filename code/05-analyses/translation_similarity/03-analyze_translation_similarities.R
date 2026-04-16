# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Analyze translation similarities
#' 
#'   This script analyzes the similarity of translations produced by different
#'   MT models. We used BERTscore to compute the similarity of translations 
#'   of the same source sentene. High scores indicate high similarity (i.e.,
#'   low discrepancies between translations' "meaning"). Low scores indicate
#'   high discrepancies between translations' "meaning".
#'   
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

## load libraries ----

library(readr, quietly=TRUE, warn.conflicts=FALSE)
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)
library(stringr, quietly=TRUE, warn.conflicts=FALSE)
library(ggplot2, quietly=TRUE, warn.conflicts=FALSE)
library(ggridges, quietly=TRUE, warn.conflicts=FALSE)
library(broom, quietly=TRUE, warn.conflicts=FALSE)

## paths ----

data_path <- file.path("data")
results_path <- file.path(data_path, "results", "translation_similarity")
utils_path <- file.path("code", "utils")

## figure setup ----

source(file.path(utils_path, "plot_setup.R"))
fig_path <- file.path("paper", "figures", "translation_similarity")
dir.create(fig_path, recursive = TRUE, showWarnings = FALSE)
save_plot <- purrr::partial(save_plot, fig.path = fig_path)

## table setup ---

source(file.path(utils_path, "table_setup.R"))
tables_path <- file.path("paper", "tables", "translation_similarity")
dir.create(fig_path, recursive = TRUE, showWarnings = FALSE)
save_kable <- purrr::partial(save_kable, dir = tables_path, overwrite = TRUE, .position = "!t")


## mappings ----

source(file.path(utils_path, "mappings.R"))


# load BERT score estimates ----

fp <- file.path(results_path, "cmp_translations_sample_bertscores.tsv")
scores <- read_tsv(fp, show_col_types = FALSE)

# # NOTE: we randomly sampled 500 texts per language and translation model pair
# count(scores, other_model, lang) |> View()

scores <- scores |>
  mutate(
    other_model = gsub("^text_mt_", "", other_model),
    language = language_iso2c_to_name[lang]
  ) 

# plot distributions by lang  (Figure 1) ----

p <- scores |>
  ggplot(
    aes(
      y = reorder(str_to_title(language), desc(language)), 
      x = bertscore_f1,
    )
  ) + 
    geom_density_ridges(
      aes(group = language, height = after_stat(density)),
      stat = "density",
      trim = TRUE,
      scale = 0.5,
      rel_min_height = 0.01,
      fill = "lightblue",
      color = NA,
      alpha = 0.6
    ) + 
    geom_boxplot(
      fill = NA, 
      width = 0.2, 
      outlier.colour = "darkgrey", outlier.size = 1/8, outlier.shape = 1/2
    ) +
    geom_text(
      data = scores |> 
        group_by(language, other_model) |> 
        summarize(bertscore_f1 = median(bertscore_f1)),
      aes(label = sprintf("%.3f", bertscore_f1)),
      nudge_y = .3,
      size = 8/.pt
    ) + 
    xlim(0.5, 1) +
    labs(
      y = NULL,
      x = "BERTScore F1" #\n(DeepL vs. open-source model translation)",
    ) + 
    facet_wrap(~mt_model_map[other_model])

p
# NOTE: missings expected because of missing translation directions of OPUS-MT

lab <- "bertscore_similarity_distributions_by_language_and_mt_model"
cap <- paste(
  "Distribution of similarities of open-source MT models' translations to DeepL translations by language and open-source MT model",
  "in sample of 500 sentences per language sampled from the CMP Translations corpus.", 
  "Translation similarity measured with BERTScore at translation pair level.",
  "Note that no OPUS-MT translation to English were obtained for Greek, Lithuanian, Norwegian, Portuguese, Romanian, and Slovenian due to translation direction limitations.",
  sprintf("\\label{fig:%s}", lab),
  collapse = " "
)
save_plot(p, fn = lab, cap = cap, w = 5.5, h = 6)


# t-tests (Figure C01) ----

#' @note: let's model average BERTscore ~ language for each open-source MT model
#'         and see how averages deviate from the grand mean

coefs <- bind_rows(
  "M2M (1.2B)" = scores |>
    filter(other_model == "m2m_100_1.2b") |>
    mutate(lang_ = str_to_title(language)) |> 
    lm(bertscore_f1 ~ lang_ - 1, data = _) |>
    broom::tidy(conf.int = TRUE) |>
    mutate(lang = sub("lang_", "", term))
  ,
  "OPUS-MT" = scores |>
    filter(other_model == "opus-mt") |>
    mutate(lang_ = str_to_title(language)) |> 
    lm(bertscore_f1 ~ lang_ - 1, data = _) |>
    broom::tidy(conf.int = TRUE) |>
    mutate(lang = sub("lang_", "", term)),
  .id = "mt_model"
)

p <- coefs |>
  ggplot(
    aes(
      y = reorder(lang, desc(lang)),
      x = estimate,
      xmin = conf.low,
      xmax = conf.high,
      label = sprintf("%.3f", estimate)
    )
  ) +
  geom_vline(
    data = scores |> 
      group_by(mt_model = mt_model_map[other_model]) |> 
      summarise(bertscore_f1 = mean(bertscore_f1, na.rm = TRUE)),
    mapping = aes(xintercept = bertscore_f1),
    color = "grey", 
    linetype = "dashed"
  ) +
  geom_linerange(position = position_dodge(0.5)) +
  geom_point(position = position_dodge(0.5), pch = 21, fill = "white", size = 1.5) +  
  geom_point(position = position_dodge(0.5), size = 0.2) + 
  geom_text(
    hjust = 0,
    nudge_x = 0.004,
    nudge_y = 0.11,
    size = 8/.pt
  ) +
  xlim(0.925, 1.0) +
  facet_grid(cols = vars(mt_model)) + 
  labs(
    y = NULL,
    x = "BERTscore F1",
  ) + 
  theme(
    plot.subtitle = element_text(size = unit(10, "pt"))
  )

p

lab <- "bertscore_similarity_averages_by_language_and_mt_model"
cap <- paste(
  "Average similarity of open-source MT models' translations to DeepL translations by language and open-source MT model", 
  "in sample 500 sentences per language sampled from the CMP Translations corpus.", 
  "Translation similarity measured with BERTScore at sentence pair level.",
  "Estimates obtained by regressing BERTScore F1 translation pair similarity measures on texts' source language indicator.",
  "Dashed vertical lines indicate the cross-language (grand mean) BERTScore for the giben open-source MT model.",
  "Note that no OPUS-MT translation to English were obtained for Greek, Lithuanian, Norwegian, Portuguese, Romanian, and Slovenian due to translation direction limitations.",
  sprintf("\\label{fig:%s}", lab),
  collapse = " "
)
save_plot(p, fn = lab, cap = cap, w = 5.5, h = 5)

