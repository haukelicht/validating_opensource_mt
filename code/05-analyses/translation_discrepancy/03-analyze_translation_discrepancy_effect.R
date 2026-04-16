# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Analyze translation ambiguity effect on classifiers' classification accuracy  
#' @author Hauke Licht
#' 
#' @description: The analyses in this R script make an attempt at addressing 
#' the following question: What impact do translation "errors" have on  
#'  classifiers' classification accuracy?
#' 
#' This is a difficult question. For one, we find in a previous analysis that 
#'  the agreement between MT models' translations is overall very high.
#' For another, translation is not deterministic in many cases, so we might have 
#'  to accept that there is no "true" translation.
#' This makes answering questions about the effect of translations "errors" 
#'  harder because if we cannot define what an error is, how can we assees
#'  what its impact is for downstream tasks?
#'  
#' So here is what we'll do:
#' 
#'  1. We start with the BERTscore similarity of open-source model translations 
#'     to DeepL translations. This allows us to quantify how (dis)similar 
#'     the open-source MT model's translations are from DeepL's translations. By 
#'     focusing on cases where the Deepl-based classifier agrees with "true" 
#'     labels, we can attribute part of the discrepancies to issues in the 
#'     open-source translations.
#'  2. Analyze the impact of translation discrepancies on downstream 
#'     classification accuracy: Is a higher discrepancy between DeepL and 
#'       open-source MT model translations associated with a higher probability 
#'       of the open-source MT-based classifier to predict a wrong label?
#'       (optional) Is this association conditional on Aya-8B's judgment that 
#'         the DeepL translation is better?
#'  Expectations: If translation quality impacts classification accuracy, 
#'     we would expect that in the subset of examples where the DeepL-based 
#'     classifier's predictions are correct, more discrepant open-source  
#'     translations make it more likely that the open-source MT-based classifers
#'     makes a classification mistake.
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
library(ggplot2, quietly=TRUE, warn.conflicts=FALSE)
library(lmtest, quietly=TRUE, warn.conflicts=FALSE)
library(sandwich, quietly=TRUE, warn.conflicts=FALSE)
library(broom, quietly=TRUE, warn.conflicts=FALSE)

## paths ----

data_path <- file.path("data", "intermediate")
results_path <- file.path("data", "results", "translation_discrepancy")
utils_path <- file.path("code", "utils")

## figure setup ----

source(file.path(utils_path, "plot_setup.R"))
fig_path <- file.path("paper", "figures", "classifier_finetuning")
dir.create(fig_path, showWarnings = FALSE, recursive = TRUE)
save_plot <- partial(save_plot, fig.path = fig_path)

## table setup ----

source(file.path(utils_path, "table_setup.R"))
tables_path <- file.path("paper", "tables", "classifier_finetuning")
dir.create(tables_path, showWarnings = FALSE, recursive = TRUE)
save_kable <- partial(save_kable, dir = tables_path, overwrite = TRUE, .position = "!t")

## mappings ----

source(file.path(utils_path, "mappings.R"))

# Model classification error as function of (extent of) translation discrepancy ----

# load predicted labels
fp <- file.path(data_path, "cmp_translations_sample_predicted_label_disagreements.tsv")
preds <- read_tsv(fp, show_col_types = FALSE)

# load BERTscore scorings
fp <- file.path(results_path, "cmp_translations_sample_label_disagreement_cases_bertscores.tsv")
cases_bertscores <- read_tsv(fp, show_col_types = FALSE)

## Analysis ----

preds |> 
  with(
    table(
      "commercial correct" = factor(commercial_is_correct, c(T, F), c("yes", "no")), 
      "open-source correct" = factor(opensource_is_correct, c(T, F), c("yes", "no"))
    )
  ) |> 
  prop.table() |> 
  round(3)

# join the classifications with the BERTscores
dat <- preds |>
  filter(commercial_is_correct | opensource_is_correct) |> 
  # subset to large M2M translations (OPUS-MT translation not implemented)
  filter(other_model == mt_model_map["m2m_100_1.2b"]) |>
  rename(qs_id = id) |> 
  left_join(
    select(cases_bertscores, other_model, qs_id, bertscore_f1)
  ) |> 
  select(
    task, language, label, 
    commercial_is_correct, opensource_is_correct, 
    bertscore_f1
  )

# # verify
# table(is.na(dat$bertscore_f1))

### averaged across tasks ----

dat <- dat |> 
  mutate(
    bertscore_f1_quintile = ntile(bertscore_f1, 5)
  ) |> 
  group_by(bertscore_f1_quintile) |> 
  mutate(
    bertscore_f1_quintile = sprintf("(%.03f, %.03f]", min(bertscore_f1), max(bertscore_f1))
  ) |> 
  ungroup()

dat |> count(commercial_is_correct, opensource_is_correct)

# accuracy of M2M-based classifiers in cases DeepL-based counterparts got all right
stats_opensource <- dat |> 
  filter(commercial_is_correct) |> 
  group_by(bertscore_f1_quintile) |> 
  group_split() |> 
  map_dfr(function(grp) {
    set.seed(1234)
    bss <- replicate(
      100, 
      {sample(grp$opensource_is_correct, nrow(grp), replace = TRUE)},
      simplify = TRUE
    )
    bss <- colMeans(bss)
    out <- c(mean(bss), quantile(bss, c(0.025, 0.975)))
    names(out) <- c("estimate", "conf.low", "conf.high")
    out <- bind_cols(distinct(select(grp, bertscore_f1_quintile)), as.list(out))
    return(out)
  })

# accuracy of DeepL-based classifiers in cases M2M-based counterparts got all right
stats_commercial <- dat |> 
  filter(opensource_is_correct) |> 
  group_by(bertscore_f1_quintile) |> 
  group_split() |> 
  # first() -> grp
  map_dfr(function(grp) {
    set.seed(1234)
    bss <- replicate(
      100, 
      {sample(grp$commercial_is_correct, nrow(grp), replace = TRUE)},
      simplify = TRUE
    )
    bss <- colMeans(bss)
    out <- c(mean(bss), quantile(bss, c(0.025, 0.975)))
    names(out) <- c("estimate", "conf.low", "conf.high")
    out <- bind_cols(distinct(select(grp, bertscore_f1_quintile)), as.list(out))
    return(out)
  })

stats <- bind_rows(
  "commercial (DeepL)" = stats_commercial,
  "open-source (M2M 1.2B)" = stats_opensource,
  .id = "mt_type"
)

p <- stats |> 
  ggplot(
    aes(
      x = bertscore_f1_quintile,
      y = estimate, ymin = conf.low, ymax = conf.high,
      color = mt_type,
      group = mt_type
    )
  ) + 
  geom_linerange(position = position_dodge(width = 0.5), show.legend = FALSE, linewidth = 0.5) +
  geom_point(position = position_dodge(width = 0.5), size = 1) + 
  scale_color_grey(start = 0.6, end = 0.2) +
  ylim(0.75, 1) +
  labs(
    y = "Accuracy",
    x = "\n Similarity between translations (BERTscore quintiles)",
    color = "underlying MT model evaluated:"
  ) + 
  theme(
    # axis.text.x = element_text(angle = 45, hjust = 1)
  )

p
#' @note: 
#'  - when similarity to DeepL translation is high, the M2M-based classifiers 
#'    get on average 94% of examples correct their DeepL-based counterparts 
#'    classified correctly
#'  - when similarity to DeepL translation is low, the M2M-based classifiers
#'    get on average only 87% of examples correct their DeepL-based counterparts
#'  - for DeepL-based classifiers, this difference is less pronounced
#'    - in texts with little to no discrepancy between DeepL and M2M translations,
#'      DeepL-based classifiers get on average 94% of examples correct ther 
#'      M2M-based counterparts classified correctly
#'    - in texts with relatively large discrepancy between DeepL and M2M 
#'      translations, this comparative accuarcy is only reduced to 89.5%
#'  - this suggest that discrepant translations that are likely associated with
#'    translation errors are more consequential for classification accuracy in 
#'    case of the M2M open-source MT model than in case of the DeepL.
#'  - this suggests that there is a stronger link between translation quality 
#'    and downstream measurement quality when using open-source MT models than
#'    when using the leading commercial MT model 

# FIGURE C08
lab <- "translation_discrepancy_effect"
cap <- paste(
  "Relation between the classifiers agreement and translation discrepancies.",
  "\\emph{Black} dots report the accuracy of M2M-based classifiers",
  "in test set examples their DeepL-based counterparts classified correctly",
  "by level of similarity between M2M (1.2B) to DeepL's translations measured with BERTscore.",
  "\\emph{Grey} dots report the accuracy of DeepL-based classifiers",
  "in test set examples their M2M-based counterparts classified correctly",
  "by level of similarity.",
  "Vertical bars indicate 95% confidence intervals (CIs).",
  "Accuracy pointe estimates and computed CIs from 100 bootstrapped accuracy estimates.",
  sprintf("\\label{fig:%s}", lab),
  collapse = " "
)

save_plot(p, fn = lab, cap = cap, w = 5, h = 3)
