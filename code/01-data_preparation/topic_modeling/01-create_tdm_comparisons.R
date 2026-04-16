# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Create term-document matrix comparisons for input-level translation
#'          similarity analysis in topic modeling study (study I)
#' @author Ronja Sczepanski, Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----


## load required libraries ----
renv::activate()

library(quanteda) # @ v3.2.3
library(quanteda.textstats) # @ v0.96
library(readtext, warn.conflicts = FALSE) # @ v0.81
library(purrr)
library(future)
plan(multisession)
library(furrr)

# For orginal Google Translate translations ----

rm(list=ls())

filenames <- list.files(
  file.path("data", "datasets", "topic_modeling", "texts"), 
  pattern="*en-translated-small.csv", 
  full.names=TRUE
) 
names(filenames) <- sub("-.*", "", basename(filenames))

#' Loading parsed text files from csv's
#' 
#' @param fp The name of the file to be loaded
#' 
#' @return A list of vectors containing the similarity measures for each document in the corpus.
loader <- function(fp) {
  content <- readtext(fp, text_field=2, header=FALSE)
  corp <- corpus(content)
  docnames <- unique(content$V1)
  rm(content)
  
  doc_corpora <- map(docnames, \(dn) corpus_subset(corp, V1 == dn))
  rm(corp)
  gc()
  
  ## Function to compare TDM composition per document
  similarizer <- function(doc.corp){ # docname <- docnames[1]
    temptdm <- dfm(tokens(doc.corp), verbose=FALSE)
    cosinesimil <- textstat_simil(temptdm, margin = "documents", method = "cosine")
    correlationsimil <- textstat_simil(temptdm, margin = "documents", method = "correlation")
    exact_match <- sum(temptdm[1,] == temptdm[2,])/length(temptdm[1,])
    unique_tr <- sum(temptdm[1,] == 0)/length(temptdm[1,])
    unique_en <- sum(temptdm[2,] == 0)/length(temptdm[1,])
    vector <- vector("numeric")
    vector <- c(vector, as.matrix(cosinesimil)[1,2])
    vector <- c(vector, as.matrix(correlationsimil)[1,2])
    vector <- c(vector, exact_match)    
    vector <- c(vector, 1-exact_match-unique_en-unique_tr)
    vector <- c(vector, unique_en)
    vector <- c(vector, unique_tr)
    vector <- c(vector, length(temptdm)/2)
    return(vector)
  }
  
  out <- future_map(
    doc_corpora, 
    similarizer, 
    .progress = TRUE, 
    .options = furrr_options(seed = TRUE, packages = c("quanteda", "quanteda.textstats", "readtext", "future", "furrr"))
  )
  return(out)
}


## Load the text data and compute similarities ----

finout <- map(filenames, loader)

## Creating data structures per language
cosine_da <- finout[1]
cosine_de <- finout[2]
cosine_es <- finout[3]
cosine_fr <- finout[4]
cosine_pl <- finout[5]
matrix_da <- t(simplify2array((cosine_da[[1]])))
matrix_de <- t(simplify2array((cosine_de[[1]])))
matrix_es <- t(simplify2array((cosine_es[[1]])))
matrix_fr <- t(simplify2array((cosine_fr[[1]])))
matrix_pl <- t(simplify2array((cosine_pl[[1]])))
cols <- c("Cosine Similarity","Correlation Similarity","% exactly matchin features (stems)","% mismatches in shared features","% unique features English text", "% Unique features translated text","Number of features (combined)")
colnames(matrix_da) <- cols
colnames(matrix_de) <- cols
colnames(matrix_es) <- cols
colnames(matrix_fr) <- cols
colnames(matrix_pl) <- cols
matrix_comb <- rbind(matrix_da,matrix_de,matrix_es,matrix_fr,matrix_pl)


## Generating a summary matrix of descriptives and TDM similarity scores per country
summarizer <- function(matr) {
  tempmatrix <- eval(as.name(matr))
  sumMatrix <- do.call(rbind, apply(tempmatrix[,1:6], 2, psych::describe))
  return(sumMatrix)
}

country <- c("matrix_da", "matrix_de", "matrix_es", "matrix_fr", "matrix_pl", "matrix_comb")
summary <- lapply(country, summarizer)

summary_da <- summary[[1]]
summary_de <- summary[[2]]
summary_es <- summary[[3]]
summary_fr <- summary[[4]]
summary_pl <- summary[[5]]
summary_comb <- summary[[6]]
rows <- c("Cosine Similarity","Correlation Similarity","Exactly matching features (stems)","Mismatches in shared features","Unique features English text", "Unique features translated text")

rownames(summary_da) <- rows
rownames(summary_de) <- rows
rownames(summary_es) <- rows
rownames(summary_fr) <- rows
rownames(summary_pl) <- rows
rownames(summary_comb) <- rows

## Saving the outputs ----

fp <- file.path("data", "datasets", "topic_modeling", "tdm", "TDMCompare_gt.RData")
if (!file.exists(fp)) {
  save.image(fp)
}

# for open-source translation with OPUS-MT -----

rm(list=setdiff(ls(), c("loader", "summarizer", "rows", "cols")))

filenames <- list.files(
  file.path("data", "datasets", "topic_modeling", "texts"), 
  pattern="*opus_europarl_small.csv", 
  full.names=TRUE
) 
names(filenames) <- sub("-.*", "", basename(filenames))

## Load the text files and compute similarities ----

finout <- map(filenames, loader)

## Creating data structures per language
cosine_da <- finout[1]
cosine_de <- finout[2]
cosine_es <- finout[3]
cosine_fr <- finout[4]
cosine_pl <- finout[5]
matrix_da <- t(simplify2array((cosine_da[[1]])))
matrix_de <- t(simplify2array((cosine_de[[1]])))
matrix_es <- t(simplify2array((cosine_es[[1]])))
matrix_fr <- t(simplify2array((cosine_fr[[1]])))
matrix_pl <- t(simplify2array((cosine_pl[[1]])))
cols <- c("Cosine Similarity", "Correlation Similarity", "% exactly matchin features (stems)", "% mismatches in shared features", "% unique features English text", "% Unique features translated text", "Number of features (combined)")
colnames(matrix_da) <- cols
colnames(matrix_de) <- cols
colnames(matrix_es) <- cols
colnames(matrix_fr) <- cols
colnames(matrix_pl) <- cols
matrix_comb <- rbind(matrix_da, matrix_de, matrix_es, matrix_fr, matrix_pl)

## Summarize the similarities ----

country <- c("matrix_da", "matrix_de", "matrix_es", "matrix_fr", "matrix_pl", "matrix_comb")
summary <- lapply(country, summarizer)

summary_da <- summary[[1]]
summary_de <- summary[[2]]
summary_es <- summary[[3]]
summary_fr <- summary[[4]]
summary_pl <- summary[[5]]
summary_comb <- summary[[6]]
rows <- c("Cosine Similarity", "Correlation Similarity", "Exactly matching features (stems)", "Mismatches in shared features", "Unique features English text", "Unique features translated text")

rownames(summary_da) <- rows
rownames(summary_de) <- rows
rownames(summary_es) <- rows
rownames(summary_fr) <- rows
rownames(summary_pl) <- rows
rownames(summary_comb) <- rows

## Saving the outputs ----

fp <- file.path("data", "datasets", "topic_modeling", "tdm", "TDMCompare_opus.RData")
if (!file.exists(fp)) {
  save.image(fp)
}
