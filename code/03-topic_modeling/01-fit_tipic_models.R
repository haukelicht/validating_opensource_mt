# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Fit topic models to human, google translate and OPUS-MT translations
#'          for all langauges in De Vries et al.'s subest of the europarl corpus
#' @author Ronja Sczepanski, Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

rm(list=ls())
renv::activate()

## load required libraries ----

library(parallel)
library(quanteda) # @ v4.3.0 (prev. 3.2.3)
library(topicmodels) # @ v0.2-17 (on Unix, depends on GNU Scientific Library "GSL")
library(readtext, warn.conflicts = FALSE) # @ v0.91

## define paths ----

data_path <- file.path("data", "datasets", "topic_modeling", "texts")
results_path <- file.path("data", "results", "topic_modeling")

## Setting global working directory, csv files are assumed to be in the subfolder "text". Please check this!

csv_files <- list.files(data_path, pattern="^[a-z]{2}_europarl_small.csv", full.names=TRUE)
names(csv_files) <- basename(csv_files)

#' Fit topic models
#'
#' @param file.paths chr, list of paths to CSV files recording translations
#' @param n.topics int, number of topics
#' @param save.to chr, path to save fitted TM objects
#' @param .test logical, enable test mode (fitting on subsample of data)
#'
#' @return A list of fitted topic models
processor <- function(file.paths, n.topics, save.to, .overwrite=FALSE, .test=FALSE) {
  stopifnot(
    "`file.paths` must specify paths to existing files" = is.character(file.paths) && length(file.paths) > 0 && !any(is.na(file.paths)), 
    "`n.topics` must be 90 or 100" = is.numeric(n.topics) && length(n.topics)==1 && !is.na(n.topics) && n.topics %in% c(90, 100), 
    "`save.to` must be an existing directory" = is.character(save.to) &&  length(save.to)==1 && !is.na(save.to) && dir.exists(save.to), 
    "`.overwrite` must be TRUE or FALSE" = is.logical(.overwrite) && length(.overwrite)==1 && !is.na(.overwrite), 
    "`.test`  must be TRUE or FALSE" = is.logical(.test) && length(.test)==1 && !is.na(.test)
  )
  
  fp <- file.path(save.to, sprintf("lda%d.rds", n.topics))
  if (file.exists(fp) && !isTRUE(.overwrite)) {
    message(sprintf("File %s already exists, skipping...", fp))
    return(readRDS(fp))
  }
  
  ## LDA Modeling function
  if (n.topics == 90) {
    modelizer <- function(fp, n.topics, burnin, iter, thin, seed, best, keep, .test){
      text <- readtext::readtext(fp, text_field=2, header=FALSE)
      if (.test) {
        set.seed(seed)
        idxs <- sample(1:nrow(text), 500)
        text <- text[idxs,]
      }
      corpus <- quanteda::corpus(text)
      
      tempdfm <- quanteda::dfm(quanteda::tokens(quanteda::corpus_subset(corpus, V3 == "en"), verbose=TRUE))
      dtm <- quanteda::convert(tempdfm, to = "tm")
      
      ldaOutEN <- topicmodels::LDA(dtm,k=n.topics, method="Gibbs", control = list(burnin = burnin, iter = iter, keep = keep, seed = seed))
      
      tempdfm <- quanteda::dfm(quanteda::tokens(quanteda::corpus_subset(corpus, V3 == "tr"), verbose=TRUE))
      dtm <- quanteda::convert(tempdfm, to = "tm")
      ldaOutGT <-topicmodels::LDA(dtm,k=n.topics, method="Gibbs", control = list(burnin = burnin, iter = iter, keep = keep, seed = seed))
      
      tempdfm <- quanteda::dfm(quanteda::tokens(quanteda::corpus_subset(corpus, V3 == "opus"), verbose=TRUE))
      dtm <- quanteda::convert(tempdfm, to = "tm")
      ldaOutOpus <- topicmodels::LDA(dtm,k=n.topics, method="Gibbs", control = list(burnin = burnin, iter = iter, keep = keep, seed = seed))
      
      ldaOutComb <- c(ldaOutEN, ldaOutGT, ldaOutOpus, fp)
      
      return(ldaOutComb)
    }
  } else if (n.topics == 100) {
    modelizer <- function(fp, n.topics, burnin, iter, thin, seed, best, keep, .test){
      text <- readtext::readtext(fp, text_field=2, header=FALSE)
      if (.test) {
        set.seed(seed)
        idxs <- sample(1:nrow(text), 500)
        text <- text[idxs,]
      }
      corpus <- quanteda::corpus(text)
      tempdfm <- quanteda::dfm(quanteda::tokens(quanteda::corpus_subset(corpus, V3 == "tr"), verbose=TRUE))
      dtm <- quanteda::convert(tempdfm, to = "tm")
      ldaOutGT <- topicmodels::LDA(dtm,k=n.topics, method="Gibbs", control = list(burnin = burnin, iter = iter, keep = keep, seed = seed))
      ldaOutComb <- c(ldaOutGT, fp)
      return(ldaOutComb)
    }
  } else {
    stop("n.topics must be either 90 or 100")
  }
  ## Uncomment the lines below to run LDA models in parallel to speed up processing, 
  ## number of parallel running functions is determined by no_cores
  ## Note: Keep in mind that in general, memory requirements for LDA modeling are usually larger than processing requirements.
  ## Therefore, parallel processing might fail due to insufficient memory when using all (or all -1) available processor cores
  
  n_cores <- parallel::detectCores()
  cl <- parallel::makePSOCKcluster(max(length(file.paths), n_cores))
  ldaOutList <- parallel::parLapply(
    cl, 
    file.paths, 
    modelizer, 
    n.topics = n.topics,
    burnin = 1000,
    iter = 300,
    thin = 100,
    seed = 1473943969,
    best = TRUE,
    keep = 50 ,
    .test = TRUE
  )
  
  parallel::stopCluster(cl)
  
  saveRDS(ldaOutList, file=fp)
}

# fit models
processor(csv_files, n.topics = 90, save.to = results_path) # add .test = TRUE for testing

rm(list=ls())
