# setup ----

# define paths ----
paper_path <- "paper"
figures_path <- file.path(paper_path, "figures")
tables_path <- file.path(paper_path, "tables")

# helper function ----
copy_artifact <- function(src, tgt, src.root, tgt.root, .ext=NULL, .overwrite=TRUE) {
  src <- file.path(src.root, ifelse(is.null(.ext), src, paste0(src, .ext)))
  tgt <- file.path(tgt.root, ifelse(is.null(.ext), tgt, paste0(tgt, .ext)))
  
  if (!file.exists(src)) {
    warning(paste("Source file does not exist:", src), call. = FALSE, immediate. = TRUE)
    invisible()
  }
  
  if (file.exists(tgt) && !.overwrite) {
    stop(paste("Target file already exists:", tgt))
  }
  
  if (!dir.exists(tgt.root)) {
    dir.create(tgt.root, recursive=TRUE)
  }
  
  file.copy(src, tgt)
}

# figures ----
figures <- read.table(file.path(paper_path, "figure_names_map.tsv"), sep="\t", col.names=c("relpath", "name"))
args <- list(src.root=figures_path, tgt.root=paper_path, .ext=".pdf", .overwrite=TRUE)
mapply(copy_artifact, figures$relpath, figures$name, MoreArgs=args)

# tables ----
tables <- read.table(file.path(paper_path, "table_names_map.tsv"), sep="\t", col.names=c("relpath", "name"))
args <- list(src.root=tables_path, tgt.root=paper_path, .ext=".tex", .overwrite=TRUE)
mapply(copy_artifact, tables$relpath, tables$name, MoreArgs=args)
