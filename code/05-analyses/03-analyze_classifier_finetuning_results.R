# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Analyze classification model finetuning experiments
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# renv::load()

## load required packages ----

library(jsonlite, quietly=TRUE, warn.conflicts=FALSE)
library(readr, quietly=TRUE, warn.conflicts=FALSE)
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)
library(tidyr, quietly=TRUE, warn.conflicts=FALSE)
library(purrr, quietly=TRUE, warn.conflicts=FALSE)
library(stringr, quietly=TRUE, warn.conflicts=FALSE)

library(future, quietly=TRUE, warn.conflicts=FALSE)
plan(multisession, workers = 8L)
library(furrr, quietly=TRUE, warn.conflicts=FALSE)

library(lmtest, quietly=TRUE, warn.conflicts=FALSE)
library(sandwich, quietly=TRUE, warn.conflicts=FALSE)
library(broom, quietly=TRUE, warn.conflicts=FALSE)
suppressPackageStartupMessages(library(texreg, quietly=TRUE, warn.conflicts=FALSE))

library(ggplot2, quietly=TRUE, warn.conflicts=FALSE)
library(patchwork, quietly=TRUE, warn.conflicts=FALSE)
library(lemon, quietly=TRUE, warn.conflicts=FALSE)

library(irr, quietly=TRUE, warn.conflicts=FALSE)
library(TOSTER, quietly=TRUE, warn.conflicts=FALSE)

library(kableExtra, quietly=TRUE, warn.conflicts=FALSE)

## paths ----

data_path <- file.path("data", "datasets", "classifier_finetuning")
results_path <- file.path("data", "results", "classifier_finetuning")
utils_path <- file.path("code", "utils")

## figure setup ----

source(file.path(utils_path, "plot_setup.R"))
fig_path <- file.path("paper", "figures", "classifier_finetuning")
dir.create(fig_path, showWarnings = FALSE, recursive = TRUE)
save_plot <- partial(save_plot, fig.path = fig_path)

## table setup ---

source(file.path(utils_path, "table_setup.R"))
tables_path <- file.path("paper", "tables", "classifier_finetuning")
dir.create(tables_path, showWarnings = FALSE, recursive = TRUE)
save_kable <- partial(save_kable, dir = tables_path, overwrite = TRUE, .position = "!t")

## mappings ----

source(file.path(utils_path, "mappings.R"))

dataset_map_latex <- str_replace_all(dataset_map, c("&" = "\\\\&")) # , "et al\\." = "\\\\emph{et al.}"))
names(dataset_map_latex) <- names(dataset_map)

tasks <- read_tsv(file.path(data_path, "dataset_tasks_overview.tsv"), show_col_types = FALSE)

tasks_overview <- tasks |>
  filter(included == "yes", appendix == "no") |> 
  rename_all(str_replace_all, pattern = c(" " = "_")) |> 
  transmute(
    experiment_name,
    dataset_id = sub("^(.+?\\d{4})_.+", "\\1", experiment_name, perl = TRUE),
    task_name = str_replace_all(experiment_name, c("^.+?\\d{4}_" = "")),
    data = escape_latex(data),
    task = escape_latex(task),
    # label_classes,
    NULL
  ) 

tasks_map <- with(tasks_overview, set_names(task, task_name))

# parse results ----

#' Parse experiment results JSON file
#' 
#' Our classifier_finetuning.py script computes the test set performance 
#'     of the fine-tuned classifier and write it to a JSON file.
#'     This R function just converts it to suitable R data objects.
#' 
#' @param fp path to JSON file
#' @param group.var.name name of categorical indicator used to compute 
#'     group-specific evaluation metrics
#'
#' @return a list object
parse_res_file <- function(fp, group.var.name = "lang") {
  
  dat <- read_json(fp)  
  
  detailed <- tibble(.rows = 1)
  detailed[[group.var.name]] <- NA
  detailed <- bind_cols(detailed, as_tibble(map(dat$bootstrapped$overall, unlist)))
  
  if (is.list(dat$bootstrapped$grouped)) {
    detailed <- bind_rows(
      detailed,
      bind_rows(map(map_depth(dat$bootstrapped$grouped, 2, unlist), as_tibble), .id = group.var.name)
    )
  }
  
  summarized <- dat$bootstrapped$summarized |> 
    map_depth(2, function(vals) as_tibble(set_names(vals, c("mean", "q025", "q975")[1:length(vals)]))) |> 
    map(bind_rows, .id = "metric") |> 
    bind_rows(.id = group.var.name) |> 
    fill(value, .direction = "up") |> 
    rename(sample_size = value) |> 
    filter(metric != "size")
  
  preds <- dat$labels[!map_lgl(dat$labels, is.null)]
  preds <- as_tibble(map(preds, unlist))
  
  args <- dat$args[!map_lgl(dat$args, is.null)]
  
  idxs <- map_lgl(args, is.list)
  args[idxs] <- map_chr(args[idxs], paste, collapse = ", ")
  
  nm <- sub("\\.json$", "", basename(fp))
  nms <- str_split(nm, "-", n = 2)[[1]]
  
  
  out <- list(
    dataset = sub("^(.+?\\d{4})_.+", "\\1", nms[1], perl = TRUE),
    task = sub("^.+?\\d{4}_(.+)", "\\1", nms[1], perl = TRUE),
    mt_model = nms[2],
    predictions = preds,
    detailed = detailed, 
    summarized = summarized, 
    duration = as_tibble(dat$duration),
    args = as_tibble(args)
  )
  
  # handle special naming of results files for CMP Translations corpus experiments
  if (grepl("^cmp_translations_sample", nm)) {
    out$dataset <- "cmp_translations_sample"
    out$task <- sub("cmp_translations_sample_([^-]+)-.+", "\\1", nm, perl = TRUE)
  }
  
  return(out)
}

res_files <- list.files(results_path, pattern = "\\.json$", full.names = TRUE)

# combine in data frame
res <- map(res_files, parse_res_file) |> 
  tibble(value = _) |> 
  unnest_wider(value) |> 
  mutate(
    mt_model = factor(mt_model, names(mt_model_map), mt_model_map)
  ) 

## inspect ----

# # overview of dataset X tasks
# print("Overview of finetuning datasets X tasks")
# count(res, dataset, task)

# overview of dataset X tasks X translation model
print("Overview of finetuning dataset X tasks X translation model")
res |> 
  count(dataset, task, mt_model) |> 
  mutate(n = "✔") |> 
  pivot_wider(names_from = mt_model, values_from = n, values_fill = "")

# describe experiment setup: get data arguments and fine-tuning hyper-parameters ----

# note: the args are invariant across translation models
hyperparmeters <- res |> 
  group_by(dataset, task) |> 
  slice(1) |> 
  ungroup() |> 
  select(1:2, args) |> 
  unnest(args) |> 
  select(
    dataset, task, label_col,
    sampling_strategy,
    downsample_train_data, downsample_minority_ratio, minority_label,
    model_name,
    class_weighting_strategy, 
    epochs, lr, training_batch_size, gradient_accumulation, warmup_ratio, weight_decay,
    early_stopping,
    eval_metric
  ) 

hyperparmeters |> 
  filter(!grepl("_w_", task)) |> 
  mutate(
    downsampling_ratio = ifelse(downsample_train_data, NA_real_, downsample_minority_ratio)
  ) |> 
  distinct(dataset, task, epochs, training_batch_size, gradient_accumulation, downsampling_ratio) |> 
  mutate(
    # apply data sets map
    dataset = factor(dataset, names(dataset_map_latex), dataset_map_latex), 
    dataset = sub("^(.+?) \\(\\d{4}\\)$", "\\\\textbf{\\1}", dataset, perl = TRUE),
    # apply tasks map
    task = factor(task, names(tasks_map), tasks_map)
  ) |> 
  arrange(dataset, task) |> 
  rename_all(~str_replace_all(str_to_title(.), c("_" = " "))) |> 
  quick_kable(
    caption = paste(
      "Number of epochs, training batch size, and gradient accumulation steps applied when fine-tuning our classifiers.",
      "We have held other hyper-parameters constant, using",
      "a learning rate of $1e^{-5}$,", 
      "a warm-up ratio of $0.05$,",
      "and a weight decay of $0.1$.",
      collapse = " "
    ),
    label = "hyperparameters",
    escape = FALSE
  ) |> 
  column_spec(2, width = "1.8in", latex_valign = "p") |> 
  collapse_rows(
    columns = 1:2,
    latex_hline = "none",
    row_group_label_position = "stack",
    row_group_label_fonts = list(
      list(
        escape = FALSE, 
        bold = FALSE, 
        latex_align = "l", 
        latex_wrap_text = TRUE,
        latex_gap_space = "3pt",
        hline_before = FALSE, 
        hline_after = FALSE, 
        extra_latex_after = "\\addlinespace",
        indent = FALSE,
        background = "#DCDCDC"
      )
    )
  ) |> 
  save_kable(
    replace.column.types = c("p" = "L", "m" = "C")
  )

# aggregate-level analyses -----

## analyze bootstrapped F1 scores -----

# get detailed bootstrap scores
res_detailed <- select(res, 1:3, detailed)

scores_df <- res_detailed |> 
  # unpack bootstrapped scores
  unnest(detailed) |>
  # reformat metrics from columns to rows
  pivot_longer(-c(1:4), names_to = "metric") |>
  # remove metrics not computed for a given dataset X task
  filter(!is.na(value)) |> 
  # keep only metrics from binary classification or class-specific for multi-class classification
  filter(grepl("(^|_)(f1|precision|recall)$", metric)) |>
  # for multi-class classifiers: split label class abbreviation from metric name
  tidyr::extract(
    metric, c("label_class", "metric"), regex = "^(.+_)?(f1|precision|recall)$"
  ) |> 
  mutate(
    # for binary classifiers: just use task name as label class name
    label_class = ifelse(label_class == "", task, label_class),
    # recode languages
    language = language_iso2c_to_name[lang]
  )

# add test sizes 
scores_df <- left_join(
  scores_df,
  select(res, 1:3, summarized) |> 
    unnest(summarized) |> 
    select(-mean, -q025, -q975, -metric) |> 
    distinct()
)

### descriptives ----

# TABLE C9: summarize bootstrapped F1 scores by overall performance by data set X task X MT model
res |> 
  select(1:3, summarized) |> 
  # discard extended language coverage tasks 
  filter(!grepl("_w_", task)) |> 
  unnest(summarized) |> 
  filter(metric %in% c("f1", "f1_macro"), lang == "overall") |> 
  transmute(
    # apply data sets map
    dataset = factor(dataset, names(dataset_map_latex), dataset_map_latex), 
    dataset = sub("^(.+?) \\(\\d{4}\\)$", "\\\\textbf{\\1}", dataset, perl = TRUE),
    # apply tasks map
    task = factor(task, names(tasks_map), tasks_map), 
    mt_model,
    value = sprintf("%.03f [%.03f,~%.03f]", mean, q025, q975)
  ) |> 
  pivot_wider(names_from = "mt_model") |> 
  select(1:3, !!unname(mt_model_map)) |> 
  arrange(dataset, task) |> 
  quick_kable(
    caption = paste(
      "Overall (cross-language) F1 scores by dataset, outcome, and translation model",
      "Values (in brackets) report average (95\\% confidence interval) of bootstrapped test set estimates.",
      "The last column reports these scores for multilingual classifiers for comparison.",
      sep = " "
    ),
    label = "overall_f1_scores",
    col.names = c("", "", unname(mt_model_map)),
    align = c(rep("l", 2), rep("c", length(mt_model_map))),
    escape = FALSE,
    longtable = TRUE
  ) |> 
  column_spec(2, width = "0.04in", latex_valign = "p") |> 
  column_spec(2, width = "1.45in", latex_valign = "p") |> 
  column_spec(3, width = "0.75in", latex_valign = "m") |> 
  column_spec(4, width = "0.85in", latex_valign = "m") |> 
  column_spec(5, width = "0.85in", latex_valign = "m") |> 
  column_spec(6, width = "0.75in", latex_valign = "m") |> 
  column_spec(7, width = "0.75in", latex_valign = "m") |> 
  column_spec(8, width = "0.8in", latex_valign = "m") |> 
  column_spec(9, width = "0.8in", latex_valign = "m") |> 
  collapse_rows(
    columns = 1:2,
    latex_hline = "none",
    row_group_label_position = "stack",
    row_group_label_fonts = list(
      list(
        escape = FALSE, 
        bold = FALSE, 
        latex_align = "l", 
        latex_wrap_text = TRUE,
        latex_gap_space = "3pt",
        hline_before = FALSE, 
        hline_after = FALSE, 
        extra_latex_after = "\\addlinespace",
        indent = FALSE,
        background = "#DCDCDC"
      )
    )
  ) |> 
  save_kable(
    replace.column.types = c("p" = "L", "m" = "C")
  )

# TABLEs C10-C14 like above but add by language
tmp <- res |> 
  select(1:3, summarized) |> 
  # discard extended language coverage tasks 
  filter(!grepl("_w_", task)) |> 
  unnest(summarized) |> 
  filter(metric %in% c("f1", "f1_macro"), lang != "overall") |> 
  transmute(
    # # apply data sets map
    dataset,
    task = factor(task, names(tasks_map), tasks_map), 
    mt_model,
    language = str_to_title(language_iso2c_to_name[lang]),
    value = sprintf("%.03f [%.03f,~%.03f]", mean, q025, q975)
  ) |> 
  arrange(dataset, task, language)

imap(dataset_map_latex, function(nm, id) { 
  #nm = "Dupont \\& Rachuj (2022)"; id = "dupont_and_rachuj_2022"
  
  tdat <- tmp |>   
    filter(dataset == id) |> 
    select(Task = task, Language = language, mt_model, value) |> 
    pivot_wider(names_from = mt_model)
  
  tdat |> 
    quick_kable(
      caption = paste(
        sprintf("Language-specific F1 scores by task and translation model in the %s dataset.", nm),
        "Values (in brackets) report averages (95\\% confidence interval) of bootstrapped test set estimates.",
        sep = " "
      ),
      label = paste0("languagewise_f1_scores_", id),
      align = c(rep("l", 2), rep("c", ncol(tdat)-2)),
      escape = FALSE,
      longtable = TRUE
    ) |> 
    collapse_rows(
      columns = 1:2,
      latex_hline = "none",
      row_group_label_position = "stack",
      row_group_label_fonts = list(
        list(
          escape = FALSE, 
          bold = FALSE, 
          latex_align = "l", 
          latex_wrap_text = TRUE,
          latex_gap_space = "3pt",
          hline_before = FALSE, 
          hline_after = FALSE, 
          extra_latex_after = "\\addlinespace",
          indent = FALSE,
          background = "#DCDCDC"
        )
      ),
      longtable_clean_cut = FALSE
    ) |> 
    kable_styling(latex_options = "repeat_header", full_width = FALSE, position = "center", font_size = 10L) |> 
    save_kable(
      replace.column.types = c("p" = "L", "m" = "C")
    )
})


# FIGUREs C2--C6 results by data set X task ~ MT model
plots <- scores_df |>
  filter(!grepl("_w_", task)) |> 
  mutate(task = factor(task, names(tasks_map), tasks_map)) |> 
  filter(is.na(lang), metric == "f1") %>%
  split(.$dataset) |> 
  # first() -> x
  map(function(x) {
    ggplot(
      data = x, 
      aes(
        y = reorder(mt_model, desc(mt_model)),
        x = value
      )
    ) +
      geom_boxplot(outlier.color = "darkgrey", outlier.size = .5) +
      lims(x = c(0, 1)) +
      lemon::facet_rep_wrap(
        ~reorder(str_wrap(str_to_sentence(task), 55), desc(task)), 
        ncol = 1,
        scales = "free_y", 
        repeat.tick.labels = TRUE
      ) + 
      labs(
        y = NULL,
        x = "F1 score",
      ) +
      theme(
        strip.text = element_text(hjust = 0, vjust = 0)
      )
  })

heights <- res |>
  select(1:2) |>
  distinct() |>
  filter(!grepl("_w_", task)) |>
  count(dataset) |>
  with(set_names(n, dataset))

tmp <- imap(plots, function(p, .id) { 
  lab <- paste0("f1_scores_", .id)
  cap <- paste(
    "Summary of fine-tuned classifiers' macro F1 scores by",
    "task (panels) and translation source (y-axis) for",
    dataset_map_latex[.id],
    "data",
    sprintf("\\label{fig:%s}", lab),
    collapse = " "
  )
  save_plot(p, fn = lab, cap = cap, w = 5, h = heights[.id]*5/4)
})

### regression analysis: effect of MT model (type) on F1 score ----

#' @description: Below we analyze whether it makes a difference for F1 scores 
#'  (a) which type of MT model you use (commercial vs. open-source), and
#'  (b) which specific MT model you use

# subset to data used in regression models
dat <- scores_df |> 
  filter(
    # remove _overall_ (cross-lingual) F1 scores for multi-class tasks 
    !is.na(lang),
    # focus on F1
    metric == "f1",
    # remove experiments with low-resource languages because the have limited MT model coverage 
    !grepl("_w_", task),
    # remove English from comparison
    language != "english"
  ) |> 
  mutate(
    # create indicator of translation model type
    mt_model_type = case_when(
      mt_model == "multilingual" ~ "multilingual",
      mt_model %in% mt_model_map[1:3] ~ "commercial",
      TRUE ~ "open-source"
    ),
    mt_model_type = factor(mt_model_type, c("commercial", "open-source", "multilingual"))
  )

# list to collect regression model results
models <- list(main = list(), appendix = list())

#### (a) by MT model type (vs. multilingual) ----

m <- lm(value ~ mt_model_type + dataset + task + language, dat)
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 
models$main[[1]] <- list(lm = m, coefs = coef_ests)
coef_ests[2:3, ]
#' main finding: using open-source MT model, on average, results in less 
#'  than one F1 score point (0.852 points on 0-100 scale) worse performance

# TABLE C15 (robustness): exclude problematic F1 scores from very small test sets
m <- lm(value ~ mt_model_type + dataset + task + language, filter(dat, sample_size >= 45))
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class)
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE)
models$appendix[["no small test sets size"]] <- list(lm = m, coefs = coef_ests)
coef_ests[2:3, ]

# TABLE X: downward bias because old Google Translate is in the comparisons for some?
m <- lm(value ~ mt_model_type + dataset + task + language, filter(dat, mt_model != mt_model_map[3]))
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 
models$appendix[["w/o old GT translations"]] <- list(lm = m, coefs = coef_ests)
coef_ests[2:3, ]
# not really (still less than 1 F1 score point)

# TABLE XX: (mainly) driven by bad performance of small M2M?
not_this_model = mt_model_map["m2m_100_418m"]
m <- lm(value ~ mt_model_type + dataset + task + language, filter(dat, mt_model != not_this_model))
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 
models$appendix[["w/o small M2M translations"]] <- list(lm = m, coefs = coef_ests)
coef_ests[2:3, ]
# yes, quite a bit; difference gets about 40-50% smaller by if excluding small M2M model from comparison

#### (b) by MT model (vs. multilingual) ----

m <- lm(value ~ mt_model + dataset + task + language, dat) 
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 
models$main[[2]] <- list(lm = m, coefs = coef_ests)
coef_ests[1:6, ]
#' findings:
#'  - OPUS-MT is the best "free" MT model
#'    - in the comparison to DeepL, it does clearly no worse than DeepL
#'      (maybe even slightly better, but not significant) 
#'    - not sure if difference to Google Translate is significant
#'  - second best is the large M2M model
#'    - using it instead of DeepL results in a only very minor reduction in 
#'      F1 scores
#'  - however, using the small M2M model results in slightly worse F1 scores
#'    - but note that we are talking about an average 1.42 F1 score points only


# TABLE C16 (robustness): driven by social media?

# Model 1
m <- dat |> 
  filter(mt_model !=  mt_model_map[3]) |> 
  mutate(socialmedia = dataset %in% c("theocharis_et_al_2016")) |> 
  lm(value ~ mt_model_type*socialmedia + task + language, data = _)
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 
models$appendix[["social media effect"]] <- list(lm = m, coefs = coef_ests)
filter(coef_ests, grepl("open-source|socialmedia", term))

# Model 2
m <- dat |> 
  filter(mt_model != mt_model_map[3]) |> 
  mutate(socialmedia = dataset %in% c("theocharis_et_al_2016")) |> 
  lm(value ~ mt_model*socialmedia + task + language, data = _)
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 
models$appendix[["social media effect, detailed"]] <- list(lm = m, coefs = coef_ests)
filter(coef_ests, grepl("Intercept|open-source|socialmedia", term))

##### create regression tables ----

# reported in paper
models$appendix[["w/o small M2M translations"]]$coefs$estimate[2]/models$main[[1]]$coefs$estimate[2]

terms <- models$main[[2]]$coefs$term
mt_model_terms <- terms[grepl("^mt_model", terms)]
mt_model_terms <- setNames(sub("^mt_model", "", mt_model_terms), mt_model_terms) 

# TABLE 6
texreg(
  l = map(models$main, "lm"),
  caption = paste(
    "OLS coefficient estimates of the effect of using open-source vs. commercial machine translation models", 
    "for translating input texts",
    "on classifiers' language-specific out-of-sample classification performance (F1 score).", 
    collapse = " "
  ) ,
  custom.note = paste(
    "\\item %stars.",
    "\\item The F1 score is measured on a scale from 0 to 1. A coefficient estimate of, for example, +0.01 (+0.001) represents an average increase of the F1 score by 0.01 (0.001), that is, one (a tenth of one) F1 score points.",
    "\\item All models include data set, task/outcome, and language fixed effects.",
    "\\item Standard errors clustered by data set, task/outcome, language, and, in case of tasks with more than two labels, by label class.",
    NULL
  ),
  label = "tab:translation_model_effects",
  file = file.path(tables_path, "translation_model_effects.tex"),
  override.coef = models$main |> map("coefs") |> map("estimate"),
  override.se = models$main |> map("coefs") |> map("std.error"),
  override.pvalues = models$main |> map("coefs") |> map("p.value"),
  stars = c(0.001, 0.01, 0.05),
  groups = list(
    # " " = 1,
    "\\emph{Type of MT model} (ref.: commercial MT model)" = 2,
    "\\emph{Translation model} (ref.: DeepL)" = 3:7
    # " " = 8
  ) ,
  custom.coef.map = c(
    list("(Intercept)" = "Intercept"),
    list("mt_model_typeopen-source" = "open-source MT model"),
    as.list(mt_model_terms),
    list(
      "mt_model_typemultilingual" = "\\vspace{3pt} multilingual classifier",
      "mt_modelmultilingual" = "\\vspace{3pt} multilingual classifier"
    )
  ),
  leading.zero = TRUE,
  single.row = TRUE,
  caption.above = TRUE,
  center = TRUE,
  digits = 3,
  dcolumn = TRUE,
  threeparttable = TRUE,
  booktabs = TRUE,
  use.packages = FALSE
) 


# TABLE C15
names(models$appendix)
idxs <- c(1, 3) 
names(models$appendix[idxs])
texreg(
  l = unname(map(models$appendix[idxs], "lm")),
  caption = paste(
    "Additional analyses of effect of using open-source vs. commercial machine translation models", 
    "for translating input texts",
    "on classifiers' language-specific out-of-sample classification performance (F1 scores).",
    paste0("Model ", seq_along(idxs), ": ", names(models$appendix[idxs]), ".", collapse = " "),
    collapse = " "
  ),
  custom.note = paste(
    "\\item %stars.",
    "\\item The F1 score is measured on a scale from 0 to 1. A coefficient estimate of, for example, +0.01 (+0.001) represents an average increase of the F1 score by 0.01 (0.001), that is, one (a tenth of one) F1 score points.",
    "\\item All models include data set, task/outcome, and language fixed effects.",
    "\\item Standard errors clustered by data set, task/outcome, language, and, in case of tasks with more than two labels, by label class.",
    NULL
  ),
  label = "tab:translation_model_effects_appendix",
  file = file.path(tables_path, "translation_model_effects_appendix.tex"),
  override.coef = models$appendix[idxs] |> map("coefs") |> map("estimate"),
  override.se = models$appendix[idxs] |> map("coefs") |> map("std.error"),
  override.pvalues = models$appendix[idxs] |> map("coefs") |> map("p.value"),
  stars = c(0.001, 0.01, 0.05),
  groups = list(
    "\\emph{Type of model} (ref.: commercial MT model)" = 1:2
  ) ,
  custom.coef.map = c(
    list(
      "mt_model_typeopen-source" = "open-source MT model",
      "mt_model_typemultilingual" = "multilingual classifier"
    ),
    list("(Intercept)" = "Intercept")
  ),
  leading.zero = TRUE,
  single.row = TRUE,
  caption.above = TRUE,
  center = TRUE,
  digits = 3,
  dcolumn = TRUE,
  booktabs = TRUE,
  threeparttable = TRUE,
  use.packages = FALSE
) 

# TABLE C16
idxs <- 4:5
texreg(
  l = unname(map(models$appendix[idxs], "lm")),
  caption = paste(
    "Effect of using open-source vs. commercial machine translation models", 
    "for translating input texts",
    "on classifiers' language-specific out-of-sample classification performance (F1 scores)",
    "in social media vs. other domains.",
    "Classifiers fine-tuned on old Google translations not included in comparison.",
    collapse = " "
  ),
  custom.note = paste(
    "\\item %stars.",
    "\\item The F1 score is measured on a scale from 0 to 1. A coefficient estimate of, for example, +0.01 (+0.001) represents an average increase of the F1 score by 0.01 (0.001), that is, one (a tenth of one) F1 score points.",
    "\\item All models include data set, task/outcome, and language fixed effects.",
    "\\item Standard errors clustered by data set, task/outcome, language, and, in case of tasks with more than two labels, by label class.",
    NULL
  ),
  label = "tab:translation_model_effects_socialmedia",
  file = file.path(tables_path, "translation_model_effects_socialmedia.tex"),
  override.coef = models$appendix[idxs] |> map("coefs") |> map("estimate"),
  override.se = models$appendix[idxs] |> map("coefs") |> map("std.error"),
  override.pvalues = models$appendix[idxs] |> map("coefs") |> map("p.value"),
  stars = c(0.001, 0.01, 0.05),
  groups = list(
    "\\emph{Type of model} (ref.: commercial MT model)" = 1:2,
    "\\emph{Translation model} (ref.: DeepL)" = 3:7,
    "\\emph{Social media vs. other domains}" = 8:15
  ) ,
  custom.coef.map = c(
    list(
      "mt_model_typeopen-source" = "open-source MT model",
      "mt_model_typemultilingual" = "multilingual classifier"
    ),
    as.list(set_names(mt_model_map, paste0("mt_model", mt_model_map))[-1]),
    list(socialmediaTRUE = "social media data"),
    list(
      "mt_model_typeopen-source:socialmediaTRUE" = "open-source MT model X social media data",
      "mt_model_typemultilingual:socialmediaTRUE" = "multilingual classifier X social media data"
    ),
    as.list(set_names(paste(mt_model_map, "X social media data"), paste0("mt_model", mt_model_map, ":socialmediaTRUE"))[-1]),
    list("(Intercept)" = "Intercept")
  ),
  leading.zero = TRUE,
  single.row = TRUE,
  caption.above = TRUE,
  center = TRUE,
  digits = 3,
  dcolumn = TRUE,
  booktabs = TRUE,
  threeparttable = TRUE,
  use.packages = FALSE
) 

#### (c) language-specific effects (interaction effects) -----

# # for which languages do we have data from multiple tasks?
# dat |> 
#   group_by(language) |> 
#   summarize(n_tasks = n_distinct(dataset, task)) |> 
#   arrange(desc(n_tasks))

high_resource_langs <- c("german", "spanish", "dutch", "french", "danish", "swedish", "italian")

tmp <- dat |> 
  filter(!mt_model %in% mt_model_map[c("google_old", "m2m_100_418m", "multilingual")]) |> 
  mutate(high_resource_lang = language %in% high_resource_langs)
  
m <- lm(value ~ mt_model_type*language + dataset + task, tmp)
cses <- vcovHC(m, type = "HC1", cluster = ~ mt_model + dataset + task + language + label_class) 
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE) 

##### export regression table ----

# TABLE C18
these_langs <- sort(unique(tmp$language))
these_langs <- these_langs[2:length(these_langs)]

texreg(
  l = m,
  caption = paste(
    "Effect of using open-source vs. commercial machine translation models",
    "for translating input texts",
    "on classifiers' language-specific out-of-sample classification performance (F1 scores)",
    "conditional on source language.",
    "Classifier fine-tuned using old Google or small M2M (418M) translations not included in comparison.",
    "The F1 score is measured on a scale from 0 to 1. A coefficient estimate of, for example, +0.01 (+0.001) represents an average increase of the F1 score by 0.01 (0.001), that is, one (a tenth of one) F1 score points.",
    "Model includes MT model, dataset and task fixed effects.",
    "Standard errors clustered by MT model, data set, task/outcome, language, and, in case of tasks with more than two labels, by label class.",
    collapse = " "
  ),
  custom.note = c("%stars."),
  label = "tab:f1_score_regression_mt_type_effects_by_language",
  file = file.path(tables_path, "f1_score_regression_mt_type_effects_by_language.tex"),
  override.coef = coef_ests$estimate,
  override.se = coef_ests$std.error,
  override.pvalues = coef_ests$p.value,
  stars = c(0.001, 0.01, 0.05),
  groups = list(
    # " " = 1,
    "\\emph{Type of model} (ref.: commercial MT model)" = 2,
    "\\emph{Language fixed effects} (ref.: Bulgarian)" = 3:(length(these_langs)+2),
    "\\emph{Language interactions effects}" = (length(these_langs)+3):(length(these_langs)*2+2)
  ),
  custom.coef.map = c(
    list("(Intercept)" = "Intercept"),
    list("mt_model_typeopen-source" = "open-source"),
    as.list(setNames(str_to_title(these_langs), paste0("language", these_langs))),
    as.list(
      setNames(
        paste(str_to_title(these_langs), "$\\times$ open-source MT model"),
        paste0("mt_model_typeopen-source:language", these_langs)
      )
    )
  ),
  leading.zero = TRUE,
  single.row = TRUE,
  caption.above = TRUE,
  center = TRUE,
  digits = 3,
  dcolumn = TRUE,
  booktabs = TRUE,
  threeparttable = FALSE,
  longtable = TRUE,
  use.packages = FALSE
)

##### plot predicted effects (Figure 3)  ----

simd <- expand.grid(
  mt_model_type = c("commercial", "open-source"),
  language = unique(tmp$language),
  dataset = unique(tmp$dataset),
  task = unique(tmp$task)
)

# Predict F1 scores using the fitted model
simd[c("pred", "lwr", "upr")] <- suppressWarnings(predict(m, newdata = simd, type = "response", interval = "confidence"))

# average across datasets and tasks
simd <- simd |> 
  group_by(mt_model_type, language) |> 
  summarize(across(pred:upr, mean), .groups = "keep") 

# Calculate the difference in predicted F1 scores between the commercial and open-source models
diffs <- simd |> 
  select(language, mt_model_type, pred) |> 
  pivot_wider(names_from = mt_model_type, values_from = pred) |> 
  transmute(language, difference = `open-source` - commercial)

simd <- left_join(simd, diffs, by = "language")

# Plotting the interaction effects
simd <- mutate(simd, low_resource_lang = !language %in% high_resource_langs) 

p <- simd |> 
  ggplot(aes(
    y = reorder(str_to_title(language), desc(language)),
    x = pred, xmin = lwr, xmax = upr,
    color = mt_model_type,
    group = mt_model_type
  )) +
    geom_linerange(position = position_dodge(0.5)) +
    # geom_point(position = position_dodge(0.5)) +
    geom_point(position = position_dodge(0.5), pch = 21, fill = "white", size = 2) +  
    geom_point(position = position_dodge(0.5), size = 0.2) + 
    geom_text(
      data = simd |> 
        group_by(language, low_resource_lang) |> 
        summarize(
          pred = max(pred), 
          difference = mean(difference)
        ) |> 
        mutate(mt_model_type = "commercial", lwr = NA_real_, upr = NA_real_),
      aes(label = sprintf("%+0.3f", difference), x = pred),  
      hjust = 0, 
      vjust = 0,
      nudge_x = 0.02,
      nudge_y = -0.1,
      size = 8/.pt,
      color = "black"
    ) +
    scale_color_manual(values = mt_model_type_color_map) +
    xlim(c(0.55, .85)) +
    facet_grid(
      rows = vars(factor(low_resource_lang, c(F, T), c("high resource", "low resource"))), 
      scales = "free_y", 
      space = "free",
      switch = "y"
    ) +
    labs(
      y = NULL,
      x = "Predicted F1-score",
      color = "MT Model Type"
    ) + 
    theme(
      plot.margin = margin(1, 0.5, 0.25, 0.25, "cm"),
      panel.spacing.y = unit(2, "lines"),
      strip.placement = "outside",
      strip.clip = "off",
      strip.text.y.left = element_text(angle=0, vjust=1, face ="bold"),
      strip.text.y = element_text(margin = margin(t=-15, r=-50)),
      strip.background = element_blank()
    )


# FIGURE 3
lab <- "f1_score_regression_mt_type_effects_by_language"
cap <- paste(
  "Predicted language-specific F1 scores by language and type of MT model.",
  "Estimates based on regression reported in Table \\ref{tab:f1_score_regression_mt_type_effects_by_language}.",
  sprintf("\\label{fig:%s}", lab),
  collapse = " "
)
save_plot(p, fn = lab, cap = cap, w = 4, h = 7)

### Equivalence tests -----

#' Generate statistics needed to compute TOST test
#'
#' @param x data frame
#' @param col column(s) to summarize
get_toster_stats <- function(x, col) {
  summarize(
    x,
    across({{col}}, list(m = mean, sd = sd, n = length), .names = "{fn}"),
    .groups = "keep"
  )
}

#' Apply TOST to data
#'
#' @param x data frame with stats for the commercial and open-source MT models 
#'     Must contain columns `mt_model`, `m`, `sd`, and `n`.
#'     Use `get_toster_stats` to compute the required stats.
#' @param commercial_models character vector of commercial MT model names
#' @param opensource_models character vector of open-source MT model names
#' @param equivalence_bound list of equivalence bounds
#'     Each element of the list should be a numeric vector of length 1 or 2, 
#'     where the first element is the lower bound and the second element is the upper bound.
#'     Defaults to `list(c(-0.01, 0.01), c(-0.02, 0.02), c(-0.03, 0.03))`
#' @param ... additional arguments passed to `TOSTER::tsum_TOST`
#'
#' @return a data frame with the results of the TOST tests
apply_tost <- function(
    x,
    commercial_models = unname(mt_model_map[1:2]),
    opensource_models = unname(mt_model_map[4:6]),
    equivalence_bound = list(
      c(-0.01, 0.01), 
      c(-0.02, 0.02), 
      c(-0.03, 0.03)
    ),
    ...
) {
  req_cols <- c("mt_model", "m", "sd", "n")
  req_cols_msg <- sprintf("`x` must have columns %s", paste0('"', req_cols, '"', collapse = ", "))
  stopifnot(
    "`x` must be a data frame with at least two rows" = is.data.frame(x) & nrow(x) >= 2,
    req_cols_msg = all(req_cols %in% colnames(x)),
    "`commercial_models` must be a character vector with at least one element and no NAs" = (
      is.character(commercial_models) & 
        length(commercial_models) > 0 & 
        !anyNA(commercial_models)
    ),
    "`opensource_models` must be a character vector with at least one element and no NAs" = (
      is.character(opensource_models) & 
        length(opensource_models) > 0 & 
        !anyNA(opensource_models)
    ),
    "`equivalence_bound` must be a list of numeric vectors with at least one element and each vector must have one or two elements" = (
      is.list(equivalence_bound) & 
        length(equivalence_bound) > 0 &
        all(sapply(equivalence_bound, is.numeric)) & 
        all(lengths(equivalence_bound) >= 1) &
        all(lengths(equivalence_bound) <= 2)
    )
  )
  
  idxs <- lengths(equivalence_bound) == 1
  equivalence_bound[idxs] <- map(equivalence_bound[idxs], \(b) sort(c(-b, b)))
  equivalence_bound[!idxs] <- map(equivalence_bound[!idxs], sort)

  # subset to relevant models
  x <- filter(x, mt_model %in% c(commercial_models, opensource_models))
  
  # create pairs
  pairings <- expand.grid(
    commercial = as.character(x$mt_model[x$mt_model %in% commercial_models]),
    opensource = as.character(x$mt_model[x$mt_model %in% opensource_models]),
    stringsAsFactors = FALSE
  )
  # stopifnot(
  #   "values in x$mt_model do not yield any valid pairings of commerical and open-source models" = nrow(pairings) > 0,
  # )
  pairings <- map(split(pairings, 1:nrow(pairings)), compose(unname, unlist, as.list))
  
  x <- split(x, x$mt_model, drop = TRUE)
  
  map_dfr(pairings, function(pair) { # pairings[[1]] -> pair
    cols <- c("m", "sd", "n")
    params <- c(as.list(x[[pair[1]]][cols]), as.list(x[[pair[2]]][cols]))
    names(params) <- paste0(names(params), rep(1:2, each = 3))
    res <- map_dfr(equivalence_bound, function(b) {
      res <- do.call(TOSTER::tsum_TOST, c(params, list(eqb = b), list(...)))
      out <- tibble::rownames_to_column(res$TOST, var = "test")
      out$bounds <- list(b)
      return(out)
    })
    out <- bind_cols(pair = paste(pair, collapse = " vs. "), as_tibble(res))
    out[c("m1", "m2")] <- map(pair, ~x[[.]]$m)
    out[c("sd1", "sd2")] <- map(pair, ~x[[.]]$sd)
    out[c("n1", "n2")] <- map(pair, ~x[[.]]$n)
    return(out)
  })
}


#' parse the output of `apply_tost` into a wide data frame
#'
#' @param x data frame as outputted by `apply_tost`
#' @param ... additional columns to include in the resulting data frame
#' @param tests character vector of tests to include in the output.
#'     Defaults to "TOST Upper"
parse_tost_result <- function(x, ..., tests = c("TOST Upper")) {
  tests <- match.arg(tests, c("TOST Upper", "TOST Lower"))
  x <- filter(x, test %in% tests)
  stopifnot(nrow(x) >= 1)
  
  if ("bounds" %in% names(x))
    x <- unnest_wider(x, bounds, names_sep="_")
  
  x <- transmute(
    x,
    ...,
    test,
    m1, m2, diff = m2-m1, 
    bounds_2, 
    res = sprintf("$t=%+0.3f$ ($p < %0.3f$)", t, p.value)
  )
  if (length(unique(x$test)) <= 1)
    x$test <- NULL
  bounds_vals <- as.character(sort(unique(x$bounds_2)))
  x <- pivot_wider(x, names_from = bounds_2, values_from = res)
  return(x)
}

#### cross-language averages ----

# overall F1 score difference grouping commercial and open-source models
tosts_overall_binary <- scores_df |>
  filter(
    metric == "f1", 
    mt_model != "multilingual",
    is.na(language) # use only (label-class-specific) cross-language F1 scores 
  ) |>
  mutate(mt_model = mt_model_type_map[mt_model]) |>
  group_by(mt_model) |>
  get_toster_stats(value) |>
  ungroup() |> 
  apply_tost(
    commercial_models = "commercial",
    opensource_models = "open-source",
    alpha = 0.05, var.equal = FALSE, hypothesis = "EQU"
  ) 

#' overall F1 score difference grouping commercial and open-source models, 
#'  excluding Google Translate (old) and M2M (418M)
tosts_overall_binary_subset <- scores_df |>
  filter(
    metric == "f1", 
    is.na(language), # use only (label-class-specific) cross-language F1 scores 
    !mt_model %in% c("multilingual", "Google Translate (old)", "M2M (418M)")
  ) |>
  mutate(mt_model = mt_model_type_map[mt_model]) |>
  group_by(mt_model) |>
  get_toster_stats(value) |>
  apply_tost(
    commercial_models = "commercial",
    opensource_models = "open-source",
    alpha = 0.05, var.equal = FALSE, hypothesis = "EQU"
  )

tosts_overall <- scores_df |>
  filter(
    metric == "f1", 
    is.na(language), # use only (label-class-specific) cross-language F1 scores 
    mt_model != "multilingual"
  ) |>
  group_by(mt_model) |>
  get_toster_stats(value) |>
  apply_tost(alpha = 0.05, var.equal = FALSE, hypothesis = "EQU") 

# TABLE C17
bind_rows(
  "all" = tosts_overall_binary,
  "subset" = tosts_overall_binary_subset,
  "all" = tosts_overall,
  .id = "what"
) |> 
  parse_tost_result(pair, what) |>
  arrange(desc(pair)) |> 
  mutate(
    pair = ifelse(what=="subset", paste0(pair, "\\,$^{a}$"), pair),
    what = NULL
  ) |> 
  quick_kable(
    caption = paste(
      "TOST equivalence tests for cross-language F1 score differences between commercial and open-source MT-based classifiers.",
      "$t$-statistics and $p$-values computed at $\\alpha = 0.05$.",
      collapse = " "
    ),
    label = "classifiers_f1_equivalence_overall",
    col.names = c(
      "Comparison", 
      "commercial", "open-source", "difference", 
      sprintf("$\\pm %0.2f$", c(0.01, 0.02, 0.03))
    ),
    align = c("l", rep("r", 6)),
    escape = FALSE
  ) |> 
  add_header_above(c(" " = 1, "Average F1 score" = 3, "Equivalence bounds" = 3)) |> 
  add_footnote(
    label = "$^{a}$ Omitting results for classifiers based on old Google Translate translations and M2M (418M).",
    notation = "none",
    escape = FALSE
  ) |> 
  save_kable()
  

#### language-specific averages ----

tosts_by_language_overall_subset <- scores_df |>
  filter(
    metric == "f1", 
    !mt_model %in% c("multilingual", "Google Translate (old)", "M2M (418M)"),
    !is.na(language),
    language != "english"
  ) |> 
  mutate(mt_model = mt_model_type_map[mt_model]) |>
  group_by(language, mt_model) |>
  get_toster_stats(value) |>
  group_by(language) |>
  group_map(
    function(x, group) {
      out <- apply_tost(
        x, 
        commercial_models = "commercial",
        opensource_models = "open-source",
        alpha = 0.05, var.equal = FALSE, hypothesis = "EQU"
      )
      return(bind_cols(group, out))
    }
  ) |>
  bind_rows()

tosts_by_language_subset <- scores_df |>
  filter(
    metric == "f1", 
    !mt_model %in% c("multilingual", "Google Translate (old)", "M2M (418M)"),
    !is.na(language),
    language != "english"
  ) |> 
  group_by(language, mt_model) |>
  get_toster_stats(value) |>
  group_by(language) |>
  group_map(
    function(x, group) {
      out <- apply_tost(x, alpha = 0.05, var.equal = FALSE, hypothesis = "EQU")
      return(bind_cols(group, out))
    }
  ) |>
  bind_rows()

# TABLE C19
tosts_by_language_overall_subset |> 
  parse_tost_result(language) |> 
  arrange(language) |> 
  mutate(language = str_to_title(language)) |>
  quick_kable(
    caption = paste(
      "TOST equivalence tests for language-specific F1 score differences between commercial and open-source MT-based classifiers.",
      "Omitting classifiers based on old Google Translate translations and M2M (418M).",
      "$t$-statistics and $p$-values computed at $\\alpha = 0.05$.",
      collapse = " "
    ),
    label = "classifiers_f1_equivalence_by_lang",
    col.names = c(
      "Language",
      "commercial", "open-source", "difference", 
      sprintf("$\\pm %0.2f$", c(0.01, 0.02, 0.03))
    ),
    align = c(rep("l", 1), rep("r", 6)),
    escape = FALSE,
    longtable = TRUE
  ) |> 
  add_header_above(c(" " = 1, "Average F1 score" = 3, "Equivalence bounds" = 3)) |> 
  save_kable()

# prediction-level analysis -----

## using prediction-level agreement -----

#' @note because the same F1 score value might be achieved with substantially 
#'  different classifications (same rate of errors but in different examples)
#' Here we focus on agreement of predicted labels of different classifiers 

preds <- select(res, 1:3, predictions) 

# create pairings with DeepL
pairings <- map(unname(mt_model_map[-1]), ~c(unname(mt_model_map[1]), .))

# iterate over all pairs of MT models
preds_agreement <- map_dfr(pairings, function(pair) {
  message(paste(pair, collapse = " vs. "))
  
  # subset predictions to relevant MT models 
  out <- preds |> 
    filter(mt_model %in% pair) |> 
    group_by(dataset, task) |> 
    filter(n_distinct(mt_model) == 2)  |> 
    group_split() |> 
    future_map_dfr(function(dat) {
      
      # get the predictions by the classifiers trained with translations from the respective MT models
      a <- dat$predictions[dat$mt_model == pair[1]][[1]]
      b <- dat$predictions[dat$mt_model == pair[2]][[1]]
      
      # split them by language
      a_grpd <- split(a$pred, a$group)
      b_grpd <- split(b$pred, b$group)
      
      # for each language, compute agreements in predicted labels
      am <- map2_dfr(
        a_grpd,
        b_grpd[names(a_grpd)],
        function(a, b) {
          f <- suppressWarnings(irr::kappam.fleiss(cbind(a, b)))
          k <- suppressWarnings(irr::kripp.alpha(rbind(a, b), method = "nominal"))
          u <- suppressWarnings(irr::agree(cbind(a, b)))
          u$irr.name <- "agreement"
          u$value <- u$value/100
          tibble(
            metric = map_chr(list(f, k, u), "irr.name"),
            method = map_chr(list(f, k, u), "method"),
            value = map_dbl(list(f, k, u), "value"),
            size = f$subjects,
            cm = list(table(a, b))
          )
        },
        .id = "lang"
      )
      am$mt_model_a <- pair[1]
      am$mt_model_b <- pair[2]
      
      return(bind_cols(distinct(dat[1:2]), am))
    }
    )
  
  return(out)
})

preds_agreement$lang <- factor(preds_agreement$lang, names(language_iso2c_to_name), language_iso2c_to_name)

# comparison of agreement within MT-based (baseline DeepL) ~ MT model
# note: the idea here is that DeepL is the best available model
dat <- preds_agreement |> 
  filter(!grepl("_w_", task)) |> 
  filter(metric == "agreement") |> 
  # here we automatically subset to datasets for which we have DeepL translations
  filter(mt_model_a == "DeepL") |> 
  filter(!mt_model_b %in% c("multilingual", "Google Translate (old)")) |> 
  filter(lang != "english") 

# NOTE: not reported
m <- lm(value ~ mt_model_b + dataset + task + lang, data = dat)
cses <- vcovHC(m, type = "HC1", cluster = ~ dataset + task + lang)
coefs <- coeftest(m, vcov = cses)
coef_ests <- broom::tidy(coefs, conf.int = TRUE)
head(coef_ests, 6)

models <- list()
models[[1]] <- list(model = m, coef_ests = coef_ests)

coef_map <- c(
  list(
    "(Intercept)" = "Intercept",
    "mt_model_bM2M (1.2B)" = "M2M (1.2B)",
    "mt_model_bM2M (418M)" = "M2M (418M)",
    "mt_model_bOPUS-MT" = "OPUS-MT"
  )
)

# TABLE C20
texreg(
  l = map(models, "model"),
  caption = paste(
    "OLS coefficient estimates of the effect of using open-source MT model instead of Google Translate",
    "on classifiers' agreement on test set examples' predicted labels realteive to DeepL-based classifiers.",
    "Google Translate-based classifiers average agreement with DeepL-based classifiers",
    "(within dataset, task, and language), shown in the intercept, used as comparison.",
    collapse = " "
  ),
  custom.note = paste(
    "\\item %stars.",
    "\\item Agreement is measured on a scale from 0 to 1.",
    "\\item All models include data set, task/outcome, and language fixed effects.",
    "\\item Standard errors clustered by data set, task/outcome, language, and, in case of tasks with more than two labels, by label class.",
    NULL
  )
  , label = "tab:label_agreement_with_deepl"
  , file = file.path(tables_path, "label_agreement_with_deepl.tex")
  , override.coef = models |> map("coef_ests") |> map("estimate")
  , override.se =  models |> map("coef_ests") |> map("std.error")
  , override.pvalues = models |> map("coef_ests") |> map("p.value")
  , stars = c(0.001, 0.01, 0.05)
  , custom.coef.map = coef_map
  , leading.zero = TRUE
  , single.row = TRUE
  , caption.above = TRUE
  , center = TRUE
  , digits = 3
  , dcolumn = TRUE
  , threeparttable = TRUE
  , booktabs = TRUE
  , use.packages = FALSE
)

## win rate analysis using prediction-level (dis)agreement -----

#' @note: When we compare two classifiers, one fine-tuned using a commercial MT
#'  model's  translations and another classifier fine-tuned using an open-source 
#'  MT model's translations, we can identify 
#'  
#'  1. cases where the two classifiers' predictions _both_ agree or _both_ disagree 
#'     with the "true" label, 
#'     and 
#'  2. cases where one of the classifiers' predictions agrees with the the 
#'     "true" label (i.e., correctly classifies) whereas the other disagrees
#'     with it (i.e., misclassifies). 
#' 
#' In the table below, cases in category 1 are on the diagonal and labeled N1 
#'  and N2. Cases in category 2 are on the off-diagonal and labeled N3 and N4.
#'  
#'                                           _open-source MT-based classifier_
#'                                                 correct     incorrect
#'    _commercial MT-based classifier_   correct     N1           N3
#'                                     incorrect     N4           N2
#'
#' Based cases on the off-diagonal (N3 and N4), we can estimate whether it is 
#'  more likely that the open-source MT-based classifier makes an errors when 
#'  the commercial MT-based classifier make none (i.e., N3 > N4?) than vice versa

#' @explanation: Here we compare two classifiers, one fine-tuned using commercial 
#'   MT model translations and another classifier that uses open-source MT model 
#'   translations, and focus on cases where the classifiers' predictions disagree but one of 
#'   the two classifiers' predicted labels is correct given the "true" labels.

preds <- select(res, 1:3, predictions) 

# create pairings with DeepL
these_models <- mt_model_map[c("opus-mt", "m2m_100_1.2b")]

### using DeepL as reference -----

pairings <- map(unname(these_models), ~c(unname(mt_model_map["deepl"]), .))

# iterate over all pairs of MT models
preds_disagreement_deepl <- map_dfr(pairings, function(pair) { 
  
  # subset predictions to relevant MT models 
  out <- preds |> 
    filter(mt_model %in% pair) |> 
    group_by(dataset, task) |> 
    filter(n_distinct(mt_model) == 2)  |> 
    group_split() |> 
    map_dfr(function(dat) { # splits[[13]] -> dat
      
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
        filter(
          # classifiers disagree
          pred_commercial != pred_other,
          # at least one of both is correct
          (pred_commercial == label | pred_other == label)
        ) |> 
        mutate(
          other_model = pair[2],
          commercial_is_correct = label == pred_commercial,
          language = language_iso2c_to_name[group]
        ) |> 
        count(
          other_model, 
          language,
          commercial_is_correct
        )
      
      return(bind_cols(distinct(dat[1:2]), out))
    }
    )
  
  return(out)
})

preds_disagreement_overall_props <- preds_disagreement_deepl |> 
  filter(language != "english") |>
  group_by(other_model, commercial_is_correct) |> 
  summarise(n = sum(n), .groups = "keep") |> 
  group_by(other_model) |>
  mutate(prop = n / sum(n)) |> 
  ungroup()

preds_disagreement_overall_props_deepl <- preds_disagreement_overall_props |> 
  group_by(other_model) |>
  group_split() |> 
  map_dfr(function(props) {
    tmp <- binom.test(props$n[props$commercial_is_correct], sum(props$n), p = 0.5)
    tmp <- tidy(tmp, conf.int = TRUE)
    return(bind_cols(props[1, 1], tmp))
  }) |> 
  mutate(across(where(is.numeric), ~round(., 3)))

preds_disagreement_overall_props_deepl

# by language
preds_disagreement_props_by_lang <- preds_disagreement_deepl |> 
  filter(language != "english") |>
  group_by(other_model, language, commercial_is_correct) |> 
  summarise(n = sum(n), .groups = "keep") |> 
  group_by(other_model, language) |>
  mutate(prop = n / sum(n)) |> 
  ungroup()

preds_disagreement_props_by_lang_deepl <- preds_disagreement_props_by_lang |> 
  group_by(other_model, language) |>
  group_split() |> 
  map_dfr(function(props) {
    tmp <- binom.test(props$n[props$commercial_is_correct], sum(props$n), p = 0.5)
    tmp <- tidy(tmp, conf.int = TRUE)
    return(bind_cols(distinct(props, other_model, language), tmp))
  }) 

# preds_disagreement_props_by_lang_deepl |> 
#   select(other_model, language, estimate, conf.low, conf.high, p.value, n = parameter) |> 
#   filter(other_model == "M2M (1.2B)") |> 
#   tail(15)

### using Google Translate as reference -----

pairings <- map(unname(these_models), ~c(unname(mt_model_map["google"]), .))

# iterate over all pairs of MT models
preds_disagreement_google <- map_dfr(pairings, function(pair) { #pair <- pairings[[2]]
  
  # subset predictions to relevant MT models 
  out <- preds |> 
    filter(mt_model %in% pair) |> 
    group_by(dataset, task) |> 
    filter(n_distinct(mt_model) == 2)  |> 
    group_split() |> 
    map_dfr(function(dat) { # splits[[13]] -> dat
      
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
        filter(
          # classifiers disagree
          pred_commercial != pred_other,
          # at least one of both is correct
          (pred_commercial == label | pred_other == label)
        ) |> 
        mutate(
          other_model = pair[2],
          commercial_is_correct = label == pred_commercial,
          language = language_iso2c_to_name[group]
        ) |> 
        count(
          other_model, 
          language,
          commercial_is_correct
        )
      
      return(bind_cols(distinct(dat[1:2]), out))
    }
    )
  
  return(out)
})


preds_disagreement_overall_props <- preds_disagreement_google |> 
  filter(language != "english") |>
  group_by(other_model, commercial_is_correct) |> 
  summarise(n = sum(n)) |> 
  group_by(other_model) |>
  mutate(prop = n / sum(n)) |> 
  ungroup()

preds_disagreement_overall_props_google <- preds_disagreement_overall_props |> 
  group_by(other_model) |>
  group_split() |> 
  map_dfr(function(props) {
    tmp <- binom.test(props$n[props$commercial_is_correct], sum(props$n), p = 0.5)
    tmp <- tidy(tmp, conf.int = TRUE)
    return(bind_cols(props[1, 1], tmp))
  }) |> 
  mutate(across(where(is.numeric), ~round(., 3)))

preds_disagreement_overall_props_google
#' @explanation: Now we compare classifiers fine-tuned using Google Translate
#'   translations to their counterparts fine-tuned useing an open-source MT 
#'   model's translations, again focusing on cases where the classifiers' 
#'   predictions disagree but one of the two classifiers' predicted labels is 
#'   correct given the "true" labels.
#'  Google Translate-based classifiers have a significantly higher win rate when 
#'   compared to M2M-based (~9%) and OPUS-MT-based (~3.8%) classifiers.

# by language
preds_disagreement_props_by_lang <- preds_disagreement_google |> 
  filter(language != "english") |>
  group_by(other_model, language, commercial_is_correct) |> 
  summarise(n = sum(n)) |> 
  group_by(other_model, language) |>
  mutate(prop = n / sum(n)) |> 
  ungroup()

preds_disagreement_props_by_lang_google <- preds_disagreement_props_by_lang |> 
  group_by(other_model, language) |>
  group_split() |> 
  map_dfr(function(props) {
    tmp <- binom.test(props$n[props$commercial_is_correct], sum(props$n), p = 0.5)
    tmp <- tidy(tmp, conf.int = TRUE)
    return(bind_cols(distinct(props, other_model, language), tmp))
  }) 

### combine and report results ----

# TABLE C20
bind_rows(
  "DeepL" = preds_disagreement_overall_props_deepl,
  "Google Translate" = preds_disagreement_overall_props_google,
  .id = "reference"
) |> 
  transmute(
    reference, 
    other_model, 
    estimate = sprintf("%0.3f [%0.3f, %0.3f]", estimate, conf.low, conf.high),
    p_value = sprintf("$p ≤ %0.3f$", p.value),
    n = parameter,
  ) |> 
  quick_kable(
    caption = paste(
      "Win rates of commercial MT-based classifier vs. open-source MT-based classifiers",
      "in test set examples where classifiers' predicted labels disagree but one of the two classifiers is correct."
    ), 
    col.names = c("commercial", "open-source", "win rate [95\\% CI]","$p$-value", "$N$"),
    label = "win_rate_estimates",
    escape = FALSE
  ) |> 
  add_header_above(c("MT model" = 1:2, " " = 3:5)) |>
  save_kable()

# pairwise comparisons ----

res_detailed <- select(res, 1:3, detailed)

mt_models <- levels(res$mt_model)
pairings <- apply(gtools::permutations(length(mt_models), 2, mt_models), 1, as.vector, simplify = FALSE)

pairwise_comparisons <- future_map(pairings, function(pairing) { 
  # stack bootstrapped eval metrics into long data frame
  res_detailed |> 
    filter(mt_model %in% pairing) |> 
    unnest(detailed) |> 
    pivot_longer(-c(1:4), names_to = "metric") |> 
    group_by(across(1:2)) |> 
    filter(n_distinct(mt_model) == 2) |> 
    ungroup() |> 
    # keep metrics from binary classification or class-specific for multi-class classification
    filter(grepl("(^|_)(f1|precision|recall)$", metric)) |> 
    # filter(metric %in% c("f1", "precision", "recall")) |> 
    mutate(
      metric = sub("^[^_]+_|_[^_]+$", "", metric),
      mt_model = factor(mt_model, pairing),
      pairing = paste(pairing, collapse = ' vs. '),
      language = language_iso2c_to_name[lang]
    )
})

dat <- pairwise_comparisons |> 
  bind_rows() |> 
  filter(
    # ignore comparisons with multilingual model
    !grepl("multilingual", pairing),
    # focus on comparisons to commercial models
    grepl("^DeepL", pairing) | grepl("^Google", pairing),
    # ignore comparisons between commercial models
    pairing != "DeepL vs. Google Translate",
    pairing != "Google Translate vs. DeepL"
  ) |> 
  # remove overall (cross-langauge) results
  filter(!is.na(lang)) |> 
  filter(
    # comparison for English non-sensical because text is always the same 
    language != "english",
    !is.na(value),
  ) |> 
  mutate(
    model = ifelse(mt_model %in% mt_model_map[1:3], "commercial", "open-source"),
  )

#' Fit a regression to estimate the difference in mean performances between
#'     models trained on same data but different text sources (e.g. Google 
#'     Translate vs. OPUS-MT translations) 
#'
#' @param x the data frame
#' @param .pairing the text source pairing
#'
#' @return a data frame of the "tidied" regression results (generated with `broom:::tidy.coeftest`)
fit_regression <- function(x, .pairing) {
  # regress bootstrapped eval metric scores on text source, using task fixed effects
  m <- lm(value ~ mt_model + task, x)
  # cluster standard errors by language and task
  cses <- vcovHC(m, type = "HC1", cluster = ~ task + language)
  coef_ests <- coeftest(m, vcov = cses)
  broom::tidy(coef_ests, conf.int = TRUE)
  tibble(focal = .pairing[2], ref = .pairing[1], broom::tidy(coef_ests, conf.int = TRUE))
}

lms <- future_map2_dfr(pairwise_comparisons, pairings, function(x, pairing) { 
  
  # discard overall (cross-language) metrics
  x <- filter(x, !is.na(x[[4]]))
  
  with_en <- map_dfr(split(x, x$metric), fit_regression, .pairing = pairing, .id = "metric")
  
  x <- filter(x, !x[[4]] %in% c("en", "eng", "english", "English"))
  without_en <- map_dfr(split(x, x$metric), fit_regression, .pairing = pairing, .id = "metric")
  
  out <- bind_rows(
    "with_en" = with_en,
    "without_en" = without_en,
    .id = "set"
  )
  
  return(out)
})

# Figure C7
p <- lms |> 
  filter(
    metric == "f1",
    grepl("^mt_model[^:]+$", term),
    set == "without_en",
    focal != "multilingual", ref != "multilingual"
  ) |> 
  ggplot(aes(y = reorder(paste(focal, "..."), desc(focal)), x = estimate, xmin = conf.low, xmax = conf.high)) + 
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = .2) + 
    geom_linerange() +
    geom_point(pch = 21, fill = "white", size = 2) +  
    geom_point(size = 0.2) + 
    lims(x = c(-.04, .04)) +
    lemon::facet_rep_wrap( # NOTE: depends on gtable::gtable_add_row_space()
      ~paste("... vs.", ref), 
      ncol = 1,
      scales = "free_y", 
      repeat.tick.labels = TRUE
    ) + 
    labs(
      x = "Mean difference in 100 bootstrapped F1 scores",
      y = NULL
    ) + 
    theme(strip.text.y = element_text(angle = 0))

p

lab <- "pairwise_comparison_effects"
cap <- paste(
  "Summary of mean differences estimated by from regressions that compare the ",
  "performances of classifiers fine-tuned using texts' translations",
  "generated with different machine translation models as input.",
  "Points (horizontal lines) indicate the mean difference (95\\% confidence interval)", 
  "in F1 scores of the Translation model named",
  "on the y-axis compared to the translation model named in the plot panels header.",
  "For example, the positive difference for the comparison ``Google Translate vs. Deepl''",
  "indicates that using DeepL instead of Google Translate to translate input texts", 
  "results in, on average, more reliably classifiers.",
  sprintf("\\label{fig:%s}", lab),
  collapse = " "
)

save_plot(p, fn = lab, cap = cap, w = 5, h = 8)
