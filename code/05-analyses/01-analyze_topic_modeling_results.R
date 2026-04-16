# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+ #
#
#' @title  Summarize the results of De Vries et al. (2018) re-analysis
#' @author Ronja Sczepanski, Hauke Licht
#
# ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+ #

# setup ----
options(digits = 3)
options(scipen = 6)

## load required libraries ----

# renv::activate()
library(readr, quietly=TRUE, warn.conflicts=FALSE)
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)
library(tidyr, quietly=TRUE, warn.conflicts=FALSE)
library(purrr, quietly=TRUE, warn.conflicts=FALSE)
library(topicmodels, quietly=TRUE, warn.conflicts=FALSE) # @ v0.2-17 (on Unix, depends on GNU Scientific Library "GSL")
library(TOSTER, quietly=TRUE, warn.conflicts=FALSE) # @ v0.8.4
library(tibble, quietly=TRUE, warn.conflicts=FALSE)

## define paths ----

data_path <- file.path("data", "datasets", "topic_modeling")
results_path <- file.path("data", "results", "topic_modeling")
utils_path <- file.path("code", "utils")

## figure setup ----

source(file.path(utils_path, "plot_setup.R"))
fig_path <- file.path("paper", "figures", "topic_modeling")
dir.create(fig_path, recursive = TRUE, showWarnings = FALSE)
save_plot <- partial(save_plot, fig.path = fig_path)

## table setup ---

source(file.path(utils_path, "table_setup.R"))
tables_path <- file.path("paper", "tables", "topic_modeling")
dir.create(tables_path, recursive = TRUE, showWarnings = FALSE)
save_kable <- partial(save_kable, dir = tables_path, overwrite = TRUE, .position = "!t")

## define mappings ----

translation_sources <- c(
  "Google Translate" = "gt",
  "OPUS-MT" = "opus-mt"
)

# use colorblind friendly colors
color_map <- setNames(c("#56B4E9", "#E69F00"), translation_sources)

# define mapping of model names to R(DS) object names 
models <- c("Google Translate", "OPUS-MT")
model_map <- setNames(models, translation_sources)

lang_map <- c(
    "da" = "Danish",
    "de" = "German",
    "es" = "Spanish",
    "fr" = "French",
    "pl" = "Polish",
  "comb" = "Total"
)
lang_map_latex <- lang_map
lang_map_latex["comb"] = "\\emph{Total}"

# input-level analyses ----

## Table 2: input similarity ----
message("generating Table 2")

tdms_gt <- new.env()
load(file.path(data_path, "tdm", "TDMCompare_gt.RData"), envir = tdms_gt)
tdms_opus <- new.env()
load(file.path(data_path, "tdm", "TDMCompare_opus.RData"), envir = tdms_opus)

a <- ls(pattern = "^summary_", envir = tdms_gt)
names(a) <- sub("summary_", "", a)
a <- a[names(lang_map_latex)]
a <- a |> 
  map(get, envir = tdms_gt) |>
  map(as_tibble) |>
  map_dfr(head, 1, .id = "lang") |> 
  select(lang, n, mean, sd, min, max) |> 
  mutate(lang = lang_map_latex[lang])

b <- ls(pattern = "^summary_", envir = tdms_opus)
names(b) <- sub("summary_", "", b)
b <- b[names(lang_map_latex)] 
b <- b |> 
  map(get, envir = tdms_opus) |>
  map(as_tibble) |>
  map_dfr(head, 1, .id = "lang") |> 
  select(lang, n, mean, sd, min, max) |> 
  mutate(lang = lang_map_latex[lang])

tmp <- left_join(a, b, by = c("lang", "n"), suffix = c("_gt", "_opus"))

tmp |> 
  quick_kable(
    caption = paste(
      "Summary statistics of cosine similarities between bag-of-words representations' onbtained from machine- and human-translated texts at document level.",
      "Columns grouped by translation model."
    ),
    label = "input_similarity",
    col.names = c("Language", "$N$", rep(c("Mean", "Std. dev.", "Min", "Max"), 2)),
    escape = FALSE
  ) |> 
    add_header_above(c(" " = 2, "Google Translate" = 4, "OPUS-MT" = 4)) |> 
    save_kable()

## Table B2: quivalence test for input similarity ----
message("generating Table B2")

# EXAMPLE
TOSTER::tsum_TOST(
  # Google Translate
  m1 = a$mean[1], 
  n1 = a$n[1],
  sd1 = a$sd[1],
  # OPUS-MT
  m2 = b$mean[1], 
  sd2 = b$sd[1],
  n2 = b$n[1],
  eqb = 0.01,
  alpha = 0.05,
  var.equal = TRUE,
  hypothesis = "EQU"
)
# NOTE: Only look at the lower bound as that is the effect that 
# matters (is b worse than a) but also save mean difference as, of course, 
# the one sided test is not a save procedure if the effects are too extreme 

# helper functions
compute_equivalence_rowwise <- function(dat1, dat2, equivalence_bound_value){
  mod_list <- list()
  for(i in 1:nrow(dat1)){
    mod_list[[dat1$lang[i]]] <- TOSTER::tsum_TOST(
      m1=dat1$mean[i], 
      n1=dat1$n[i], 
      sd1=dat1$sd[i], 
      m2=dat2$mean[i], 
      n2=dat2$n[i], 
      sd2 = dat2$sd[i], 
      eqb = equivalence_bound_value, alpha = 0.05, var.equal=TRUE, hypothesis = "EQU"
    )
  }
  return(mod_list)
} 

transforming_equivalence_results <- function(dat, equiv.bound){
  dat <- dat |> 
    map("TOST") |> 
    map(~rownames_to_column(.x, var = "test")) |> 
    map_dfr(as.data.frame, .id = "lang")
  
  dat$significant <- ifelse(dat$p.value<=0.05, "yes", "no")
  
  out <- dat |> 
    pivot_wider(names_from = test, values_from = c(t, SE, df, p.value, significant)) |> 
    select(c(lang, `p.value_t-test`, `p.value_TOST Upper`, `t_TOST Upper`)) |> 
    mutate(!!paste0("value_", as.character(equiv.bound)) :=sprintf("$t=%+0.3f$ ($p < %0.3f$)", `t_TOST Upper`, `p.value_TOST Upper`)) |> 
    select(-c(`p.value_TOST Upper`, `t_TOST Upper`))
  
  return(out)
}

tmp <- compute_equivalence_rowwise(a, b, 0.01)
tmp_2 <- compute_equivalence_rowwise(a, b, 0.02)
tmp_3 <- compute_equivalence_rowwise(a, b, 0.03)

tmp <- transforming_equivalence_results(tmp, 0.01)
tmp_2 <- transforming_equivalence_results(tmp_2, 0.02)
tmp_3 <- transforming_equivalence_results(tmp_3, 0.03)

tmp$difference <- a$mean - b$mean

tmp <- tmp |> 
  left_join(select(tmp_2, -c(`p.value_t-test`)), by = "lang") |>
  left_join(select(tmp_3, -c(`p.value_t-test`)), by = "lang") |> 
  select(c(lang, difference, `p.value_t-test`, value_0.01, value_0.02, value_0.03))

tmp |> 
  quick_kable(
    caption = paste(
      "$t$-tests and equivalence bounds for difference between commercial and open-source machine translation-based LDA topic models",
      "relative to benchmark LDA model fitted to human expert translations."
    ),
    label = "equivalence_input_similarity",
    col.names = c("Language", "Difference", "$p$-value", "$\\pm 0.01$", "$\\pm 0.02$", "$\\pm 0.03$"),
    align = c("l", "r", "r", "r", "r", "r"),
    escape = FALSE
  ) |>
  add_header_above(c(" " = 1, "$t$-Test" = 2, "Equivalence bounds" = 3)) |> 
  save_kable()

## Figure B1: input similarity (Figure 3 in De Vries et al., 2018) ----
message("generating Figure B1")

a <- ls(pattern = "^matrix_[a-z]{2}$", envir = tdms_gt)
names(a) <- sub("matrix_", "", a)
a <- a |>
  map(get, envir = tdms_gt) |>
  map_dfr(as_tibble, .id = "lang")

b <- ls(pattern = "^matrix_[a-z]{2}$", envir = tdms_opus)
names(b) <- sub("matrix_", "", b)
b <- b |> 
  map(get, envir = tdms_opus) |>
  map_dfr(as_tibble, .id = "lang")

p_dat <- bind_rows("gt" = a[1:2], "opus-mt" = b[1:2], .id = "translation_source")

p <- p_dat |> 
  mutate(lang = factor(lang, names(lang_map), lang_map)) |> 
  ggplot(
    aes(x = `Cosine Similarity`,
        y = reorder(lang, desc(lang)),
        fill = translation_source)
  ) + 
  geom_boxplot(linewidth = 1/3, outlier.size = .01, outlier.alpha = .25) + 
  xlim(0:1) +
  scale_fill_manual(
    name = "MT model:",
    breaks = names(color_map),
    labels = model_map,
    values = color_map
  ) + 
  guides(
    fill = guide_legend(label.hjust = 0, label.vjust = .5, override.aes = list(shape = 15, color = NA, size = 12/.pt))
  ) +
  labs(y = NULL)

cap <- paste(
  "Distribution of cosine similarities between bag-of-words representations' onbtained from machine- and human-translated texts at document level.",
  "\\label{fig:input_similarity}",
  collapse = " "
)

save_plot(p, fn = "input_similarity", cap = cap, w = 5, h = 3)

## Table B1: token overlap (Figure 4 in De Vries et al., 2018) -----
message("generating Table B1")

tmp <- read_rds(file.path(results_path, "lda90.rds"))
names(tmp) <- tmp |> map_chr(last) |> basename() |> substr(1, 2)
names(tmp) <- lang_map[names(tmp)]
tmp <- map(tmp, `[`, 1:3)

tmp <- map_dfr(tmp, function(x) {
  data.frame(
    human = length(x[[1]]@terms),
    gt = length(x[[2]]@terms),
    gt_overlap = length(intersect(x[[1]]@terms, x[[2]]@terms)),
    opus = length(x[[3]]@terms),
    opus_overlap = length(intersect(x[[1]]@terms, x[[3]]@terms))
  )
}, .id = "lang")

tmp <- tmp |>
  mutate(
    prop_overlap_gt = gt_overlap/human,
    prop_overlap_opus = opus_overlap/human
  )

tmp |>
  select(lang, human, gt, opus, contains("prop_")) |>
  quick_kable(
    caption = paste(
      "Number of tokens in topic models' vocabulary and share of tokens in machine translation-based topic models",
      "that overlap with tokens in vocabulary of topic model fitted to human experts' translations."
    ),
    col.names = c("Language", "Experts", rep(model_map, 2)),
    label = "token_overlap",
    align = c("l", rep("r", 5))
  ) |>
  add_header_above(c(" " = 1, "$N$ tokens" = 3, "Token overlap" = 2), escape = FALSE) |>
  save_kable()

# output-level analyses (topic model comparison) ----

top_gt <- new.env()
load(file.path(results_path, "comparison90_gt.RData"), envir = top_gt)

top_opus <- new.env()
load(file.path(results_path, "comparison90_opus.RData"), envir = top_opus)

## Table 3: topic proportions similarity  (Table 3 in De Vries et al., 2018) ----
message("generating Table 3")

a <- ls(pattern = "^doc2DocDistrCor[A-Z]{2}$", envir = top_gt)
names(a) <- tolower(sub("doc2DocDistrCor", "", a))
a <- a |> 
  map(get, envir = top_gt) |>
  map(as.vector) |> 
  map_dfr(tibble::enframe, name = "doc", .id = "lang")

b <- ls(pattern = "^doc2DocDistrCor[A-Z]{2}$", envir = top_opus)
names(b) <- tolower(sub("doc2DocDistrCor", "", b))
b <- b |> 
  map(get, envir = top_opus) |>
  map(as.vector) |> 
  map_dfr(tibble::enframe, name = "doc", .id = "lang")

tmp <- bind_rows(
  "gt" = a,
  "opus-mt" = b,
  .id = "translation_source"
) |> 
  mutate(lang = factor(lang, names(lang_map), lang_map))

tmp |> 
  group_by(translation_source, lang) |> 
  summarise(
    n = n(),
    mean = mean(value),
    sd = sd(value),
    sums = list(range(value))
  ) |> 
  unnest_wider(sums, names_sep = "_") |> 
  pivot_wider(names_from = translation_source, values_from = mean:sums_2) |> 
  select(lang, n, contains("gt"), contains("opus")) |> 
  quick_kable(
    caption = paste(
      "Summary statistics of correlations between document-level topic proportion estimates obtained from machine- and human-translated texts.",
      "Columns grouped by translation model."
    ),
    label = "topic_proportion_similarity",
    col.names = c("Language", "$N$", rep(c("Mean", "Std. dev.", "Min", "Max"), 2)),
    escape = FALSE
  ) |> 
  add_header_above(c(" " = 2, "Google Translate" = 4, "OPUS-MT" = 4)) |> 
  save_kable()  

## Figure B2: topic proportions similarity  (Figure 5 in De Vries et al., 2018) ----
message("generating Figure B2")

p <- tmp |> 
  ggplot(
    aes(x = value, y = reorder(lang, desc(lang)), fill = translation_source)
  ) + 
  geom_boxplot(linewidth = 1/3, outlier.size = .01, outlier.alpha = .25) +
  xlim(-.1, 1) +
  scale_fill_manual(
    name = "MT model:",
    breaks = names(color_map),
    labels = model_map,
    values = color_map
  ) + 
  guides(
    fill = guide_legend(label.hjust = 0, label.vjust = .5, override.aes = list(shape = 15, color = NA, size = 12/.pt, alpha = 1))
  ) +
  labs(
    y = NULL,
    x = "Correlation between machine- and human-translated documents"
  )

cap <- paste(
  "Similarity of document-level topic proportion estimates.",
  "\\label{fig:topic_proportion_similarity}",
  collapse = " "
)

save_plot(p, fn = "topic_proportion_similarity", cap = cap, w = 5, h = 3)

## Figure 1: topic prevalence similarity (Figure 6 in De Vries et al., 2018) ----
message("generating Figure 1")

a <- ls(pattern = "^topic2TopicDistrCor[A-Z]{2}$", envir = top_gt) 
names(a) <- tolower(sub("topic2TopicDistrCor", "", a))
a <- a |> 
  map(get, envir = top_gt) |>
  map(as.vector) |> 
  map_dfr(tibble::enframe, name = "topic", .id = "lang")

b <- ls(pattern = "^topic2TopicDistrCor[A-Z]{2}$", envir = top_opus)
names(b) <- tolower(sub("topic2TopicDistrCor", "", b))
b <- b |>   
  map(get, envir = top_opus) |>
  map(as.vector) |> 
  map_dfr(tibble::enframe, name = "topic", .id = "lang")

tmp <- bind_rows(
    "gt" = a,
    "opus-mt" = b,
    .id = "translation_source"
  ) |> 
  mutate(lang = factor(lang, names(lang_map), lang_map))

p <- tmp |> 
  ggplot(
    aes(x = value, y = reorder(lang, desc(lang)), fill = translation_source, height = after_stat(density))
  ) + 
  geom_density_ridges(stat = "binline", bins = 20, scale = 0.85, draw_baseline = FALSE, color = NA, alpha = 2/3) +
  xlim(-.1, 1) +
  scale_fill_manual(
    name = "MT model:",
    breaks = names(color_map),
    labels = model_map,
    values = color_map
  ) + 
  guides(
    fill = guide_legend(label.hjust = 0, label.vjust = .5, override.aes = list(shape = 15, color = NA, size = 12/.pt, alpha = 1))
  ) +
  labs(
    y = NULL,
    x = "Correlation between machine- and human-translated documents"
  )

cap <- paste(
  "Similarity of corpus-level topical prevalence.",
  "\\label{fig:topic_prevalence_similarity}",
  collapse = " "
)

save_plot(p, fn = "topic_prevalence_similarity", cap = cap, w = 5, h = 3)

# 
# p_dat |> 
#   group_by(translation_source, lang) |> 
#   summarise(n = n(), mean = mean(value), sd = sd(value), sums = list(range(value))) |> 
#   unnest_wider(sums, names_sep = "_") |> 
#   pivot_wider(names_from = translation_source, values_from = mean:sums_2) |> 
#   select(lang, n, contains("gt"), contains("opus")) |> 
#   quick_kable(
#     caption = paste(
#       "Summary statistics of correlations between corpus-level topic prevalence estimates obtained from machine- and human-translated texts.",
#       "Columns grouped by translation model."
#     )
#     , label = "topic_prevalence_similarity"
#     , col.names = c("Language", "$N$", rep(c("Mean", "Std. dev.", "Min", "Max"), 2))
#     , escape = FALSE
#   ) |> 
#   add_header_above(c(" " = 2, "Google Translate" = 4, "OPUS-MT" = 4)) |> 
#   save_kable()  

## Figure B3: topic content similarity (Figure 7 in De Vries et al., 2018) ----
message("generating Figure B3")

a <- ls(pattern = "^topic2TopicSimilCor[A-Z]{2}$", envir = top_gt)
names(a) <- tolower(sub("topic2TopicSimilCor", "", a))
a <- a |>   
  map(get, envir = top_gt) |>
  map(as.vector) |> 
  map_dfr(tibble::enframe, name = "topic", .id = "lang")

b <- ls(pattern = "^topic2TopicSimilCor[A-Z]{2}$", envir = top_opus) 
names(b) <- tolower(sub("topic2TopicSimilCor", "", b))
b <- b |>   
  map(get, envir = top_opus) |>
  map(as.vector) |> 
  map_dfr(tibble::enframe, name = "topic", .id = "lang")

p_dat <- bind_rows(
    "gt" = a,
    "opus-mt" = b,
    .id = "translation_source"
  ) |> 
  mutate(lang = factor(lang, names(lang_map), lang_map))

p <- p_dat |> 
  ggplot(
    aes(x = value, y = reorder(lang, desc(lang)), fill = translation_source, height = after_stat(density))
  ) + 
  geom_density_ridges(stat = "binline", bins = 20, scale = 0.85, draw_baseline = FALSE, color = NA, alpha = 2/3) +
  xlim(-.1, 1) +
  scale_fill_manual(
    name = "MT model:",
    breaks = names(color_map),
    labels = model_map,
    values = color_map
  ) + 
  guides(
    fill = guide_legend(label.hjust = 0, label.vjust = .5, override.aes = list(shape = 15, color = NA, size = 12/.pt, alpha = 1))
  ) +
  labs(
    y = NULL,
    x = "Correlation between machine- and human-translated documents"
  )

cap <- paste(
  "Comparisong of estimated topics' content between models based on human- and machine-translated texts.",
  "\\label{fig:topic_content_similarity}",
  collapse = " "
)

save_plot(p, fn = "topic_content_similarity", cap = cap, w = 5, h = 3)

