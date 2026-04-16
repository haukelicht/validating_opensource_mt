# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Replicate analyses reported in "Validating open-source machine 
#'          translation for quantitative text analysis"
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

options(renv.verbose = FALSE)
library(renv)
renv::load(".")

#' Run script and log output and messages
#'
#' @param name Name of the script in 
#' @param project 
#'
#' @return
#' @export
#'
#' @examples
run_script <- function(name, .dir, .project=".") {
  script <- file.path(.dir, name)
  msg <- paste0("Running script: ", script)
  message(msg)
  writeLines(msg, con)
  script <- renv:::renv_path_normalize(script, mustWork = TRUE)
  renv:::renv_scope_wd(.project)
  output <- system2(renv:::R(), c("-s", "-f", renv:::renv_shell_path(script)), stdout = TRUE, stderr = TRUE)
  cat(output, file = con, sep = "\n")
}

log_file <- "replication_run.log"
if (file.exists(log_file))
  file.remove(log_file)
con <- file(log_file, open = "a")

analyses_path <- file.path("code", "05-analyses")

run_script("01-analyze_topic_modeling_results.R", analyses_path)
run_script("02-describe_finetuning_datasets.R", analyses_path)
run_script("03-analyze_classifier_finetuning_results.R", analyses_path)
run_script("03-analyze_translation_similarities.R", file.path(analyses_path, "translation_similarity"))
run_script("03-analyze_translation_discrepancy_effect.R", file.path(analyses_path, "translation_discrepancy"))

close(con)

# remove any side effects
if (file.exists("Rplots.pdf"))
  file.remove("Rplots.pdf")

# rename and move figures and tables
source(file.path(analyses_path, "rename_figures_and_tables.R"))

