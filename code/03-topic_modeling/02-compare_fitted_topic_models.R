# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Compare topic models fitted to different translations of same 
#'          input texts
#' @author Ronja Sczepanski, Hauke Licht
#' @note   Code largely based on Erik De Vries' original implementation for 
#'          their 2018 Political Analysis publication, see 
#'          https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VKMY6N
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #


# setup ----

renv::activate()

## load required libraries ----

library(topicmodels)
library(parallel)
library(gtools)

## define paths ----

data_path <- file.path("data", "results", "topic_modeling")

# compare topic models ----

## main comparison function ----


#' Take RDS file recording topic model fits and compare them for each language
#'
#' @param input.file RDS file recording topic model (TM) fits
#' @param output.file RData file path to store results
#' @param n.topics number of topics, default is 90
#' @param ht.idx index of list element in `input.file` that records human translation-based TM fits
#' @param mt.idx index of list element in `input.file` that records machine translation-based TM fits
#'
#' @return NULL
comparizer <- function(input.file, output.file, n.topics=90, ht.idx=1, mt.idx=2) {
  ## Settings concerning the output of the processing step
  ## Define the number of n.topics loaded for the machine-translated data.
  ## When set to 90, comparisons are made between 90-topic gold standard and 90-topic machine translated models
  ## When set to 100, comparisons are made between 90-topic gold standard and 100-topic machine-translated models
  
  ## Loading LDA models
  ## lda90.RData contains 90-topic models for both gold standard and machine-translated data
  ## lda100.RData only contains the 100-topic models for the machine-translated data
  ## When not making comparisons with the 100-topic MT model, lines 53 through 59 can be commented out
  ldaOut<- readRDS(input.file)
  ldaOutDA <- c(ldaOut[[1]][[ht.idx]],ldaOut[[1]][[mt.idx]])
  ldaOutDE <- c(ldaOut[[2]][[ht.idx]],ldaOut[[2]][[mt.idx]])
  ldaOutES <- c(ldaOut[[3]][[ht.idx]],ldaOut[[3]][[mt.idx]])
  ldaOutFR <- c(ldaOut[[4]][[ht.idx]],ldaOut[[4]][[mt.idx]])
  ldaOutPL <- c(ldaOut[[5]][[ht.idx]],ldaOut[[5]][[mt.idx]])
  langs <- c("DA","DE","ES","FR","PL") # NOTE hard-coding languages here
  rm(ldaOut)
  gc()
  
  ## Comparison function to compare gold standard and machine-translated LDA topic models
  topicComparison <- function(lang, column) { # lang = "DA"
    
    print(paste("Starting ", lang))
    ldaOutEN.terms <- as.matrix(terms(eval(parse(text=paste("ldaOut",lang,sep="")))[[1]], 50))
    ldaOutEN.topicprobs <- as.matrix(eval(parse(text=paste("ldaOut",lang,sep="")))[[1]]@gamma)
    ldaOutEN.wordprobs <- as.matrix(t(posterior(eval(parse(text=paste("ldaOut",lang,sep="")))[[1]])$terms))
    ldaOutTR.terms <- as.matrix(terms(eval(parse(text=paste("ldaOut",lang,sep="")))[[column]], 50))
    ldaOutTR.topicprobs <- as.matrix(eval(parse(text=paste("ldaOut",lang,sep="")))[[column]]@gamma)
    ldaOutTR.wordprobs <- as.matrix(t(posterior(eval(parse(text=paste("ldaOut",lang,sep="")))[[column]])$terms))
    
    ## Creating a list of shared stems to match between gold standard and machine-translated models
    wordsEN <- as.list(rownames(ldaOutEN.wordprobs))
    wordsTR <- as.list(rownames(ldaOutTR.wordprobs))
    wordsOverlap <- intersect(wordsEN,wordsTR)
    
    ## Creating temporary matrix for probabilities of words over n.topics for each model
    ENTemp <- ldaOutEN.wordprobs
    TRTemp <- ldaOutTR.wordprobs
    
    ## Removing all words that are unique to their specific model
    ENTemp <- ENTemp[rownames(ENTemp) %in% wordsOverlap, ]
    TRTemp <- TRTemp[rownames(TRTemp) %in% wordsOverlap, ]
    
    ## Ordering remaining words alphabetically, so rows are aligned
    ENTemp <- ENTemp[ order(row.names(ENTemp)), ]
    TRTemp <- TRTemp[ order(row.names(TRTemp)), ]
    
    ## For each overlapping stem, identify for both the gold standard and machine-translated model on which topic the stem loads highest
    topicLinker <- function(word, en.wps, tr.wps) {
      en <- which.max(en.wps[word,])
      tr <- which.max(tr.wps[word,])
      comb <- paste(en,",",tr,sep="")
      return(comb)
    }
    
    n_cores <- parallel::detectCores()-2
    cl <- parallel::makeCluster(n_cores)
    topicComb <- parallel::parLapply(cl, wordsOverlap, topicLinker, en.wps = ldaOutEN.wordprobs, tr.wps = ldaOutTR.wordprobs)
    parallel::stopCluster(cl)
    
    ## Create table of most frequently occurring topic pairs over all stems
    topicComb <- unlist(topicComb)
    topicCombTab <- sort(table(topicComb), decreasing=TRUE)
    topicComb <- rownames(topicCombTab)
    
    topicEN <- vector("character")
    topicTR <- vector("character")
    
    ## Getting list of unique topic pairs (to make sure a topic is not assigned to more than 1 topic in the other model)
    splitter <- function(tc) {
      temp <- strsplit(tc, ",",fixed=TRUE)
      en <- temp[[1]][1]
      tr <- temp[[1]][2]
      if (!en %in% topicEN && !tr %in% topicTR) {
        topicEN <<- c(topicEN, en)
        topicTR <<- c(topicTR, tr)
        return(paste(en,",",tr,sep=""))
      }
    }
    
    topicUnique <- unlist(lapply(topicComb, splitter))
    
    ## Checking unassigned topic pairs and listing them
    notAssignedEN <- setdiff(as.character(1:n.topics), topicEN)
    notAssignedTR <- setdiff(as.character(1:n.topics), topicTR)
    topicUnique <- gtools::mixedsort(topicUnique)
    
    ## Ordering the data based on the gold standard topic position, to get list of how machine-translated n.topics need to be reordered
    order <- function(tu) {
      temp <- strsplit(tu, ",",fixed=TRUE)
      return(as.numeric(temp[[1]][2]))
    }
    TROrder <- unlist(lapply(topicUnique,order))
    TROrder <- c(notAssignedTR,TROrder)
    
    ## Reordering the TR dataset to make the n.topics align with the EN dataset
    ## Removing unmatched n.topics from both the TR and EN datasets, for TR by placing those n.topics as the first ones in the matrix
    ## and then removing them, for EN by just taking the indices of the appropriate n.topics
    colnames(ldaOutEN.terms) <- colnames(ldaOutEN.wordprobs) <- colnames(ldaOutEN.topicprobs) <- 1:n.topics
    if(length(notAssignedEN) > 0){
      ldaOutEN.wordprobs <- ldaOutEN.wordprobs[,-as.numeric(notAssignedEN)]
      ldaOutEN.topicprobs <- ldaOutEN.topicprobs[,-as.numeric(notAssignedEN)]
      ldaOutEN.terms <- ldaOutEN.terms[,-as.numeric(notAssignedEN)]
      ENTemp <- ENTemp[, -as.numeric(notAssignedEN)]
    }
    
    
    colnames(ldaOutTR.terms) <- colnames(ldaOutTR.topicprobs) <- colnames(ldaOutTR.wordprobs) <- 1:n.topics
    ldaOutTR.topicprobs <- ldaOutTR.topicprobs[,TROrder]
    ldaOutTR.terms <- ldaOutTR.terms[,TROrder]
    ldaOutTR.wordprobs <- ldaOutTR.wordprobs[,TROrder]
    TRTemp <- TRTemp[,TROrder]
    if(length(notAssignedTR) > 0){
      ldaOutTR.wordprobs <- ldaOutTR.wordprobs[,-c(seq(1,length(notAssignedTR),1))]
      ldaOutTR.topicprobs <- ldaOutTR.topicprobs[,-c(seq(1,length(notAssignedTR),1))]
      ldaOutTR.terms <- ldaOutTR.terms[,-c(seq(1,length(notAssignedTR),1))]
      TRTemp <- TRTemp[,-c(seq(1,length(notAssignedTR),1))]
    }
    
    ## Assigning various results as global variables in order to save them
    assign(paste("ldaOutTR.terms",lang,sep=""), ldaOutTR.terms, envir=globalenv())
    assign(paste("ldaOutEN.terms",lang,sep=""), ldaOutEN.terms, envir=globalenv())
    assign(paste("ldaOutTR.topicprobs",lang,sep=""), ldaOutTR.topicprobs, envir=globalenv())
    assign(paste("ldaOutEN.topicprobs",lang,sep=""), ldaOutEN.topicprobs, envir=globalenv())
    assign(paste("ldaOutTR.wordprobs",lang,sep=""), ldaOutTR.wordprobs, envir=globalenv())
    assign(paste("ldaOutEN.wordprobs",lang,sep=""), ldaOutEN.wordprobs, envir=globalenv())
    
    assign(paste("doc2DocDistrCor",lang,sep=""), proxy::simil(ldaOutEN.topicprobs, ldaOutTR.topicprobs, method="correlation", by_rows=TRUE, pairwise=TRUE), envir=globalenv()) 
    assign(paste("doc2DocDistrCos",lang,sep=""), proxy::simil(ldaOutEN.topicprobs, ldaOutTR.topicprobs, method="cosine", by_rows=TRUE, pairwise=TRUE), envir=globalenv())
    assign(paste("topic2TopicDistrCor",lang,sep=""), proxy::simil(ldaOutEN.topicprobs, ldaOutTR.topicprobs, method="correlation", by_rows=FALSE, pairwise=TRUE), envir=globalenv()) 
    assign(paste("topic2TopicDistrCos",lang,sep=""), proxy::simil(ldaOutEN.topicprobs, ldaOutTR.topicprobs, method="cosine", by_rows=FALSE, pairwise=TRUE), envir=globalenv())
    assign(paste("topic2TopicSimilCor",lang,sep=""), proxy::simil(ENTemp, TRTemp, method="correlation", by_rows=FALSE, pairwise=TRUE), envir=globalenv()) 
    assign(paste("topic2TopicSimilCos",lang,sep=""), proxy::simil(ENTemp, TRTemp, method="cosine", by_rows=FALSE, pairwise=TRUE), envir=globalenv()) 
    return(NULL)
  }
  
  lapply(langs, topicComparison, column = ifelse(n.topics==90, 2, 3))
  
  ## Removing the LDA topic models from the environment before saving the results
  rm(ldaOutDA, ldaOutDE, ldaOutES, ldaOutFR, ldaOutPL)
  
  save.image(file = output.file)
}

fitted_objects_file <- file.path(data_path, "lda90.rds")

# RDS object in `fitted_objects_file` is a list of lists of `topicmodels::LDA` fits. 
#  - First level records languages' topic model fits
# - second level records models fitted to (i) human, (ii) Google Translate, and (iii) OPUS-MT translations

# begin with comparison between models fitted to human and Google Translate translations
fp <- file.path("comparison90_gt.RData")
comparizer(fitted_objects_file, output.file=fp, n.topics=90, ht.idx=1, mt.idx=2)

# now comparison between models fitted to human and open-source OPUS-MT translations
fp <- file.path("comparison90_opus.RData")
comparizer(fitted_objects_file, output.file=fp, n.topics=90, ht.idx=1, mt.idx=3)
