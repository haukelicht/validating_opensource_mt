# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Setup `renv` for replicating the analyses reported in manuscript
#'          "Validating open-source machine translation for quantitative text 
#'           analysis"
#' @author Hauke Licht
#' @note   run this script from the replication folder's root path 
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# install renv 1.0.10 (if needed)
installed_pkgs <- row.names(installed.packages())
if (!"renv" %in% installed_pkgs || packageVersion("renv") != "1.0.10") {
  if (!"devtools" %in% installed_pkgs)
    install.packages("devtools", type = "binary")
  devtools::install_version("renv", version = "1.0.10")
}

# NOTE: 
renv::init(project = ".", bare = TRUE, load = TRUE, restart = FALSE)

# if renv lock already exists, ...
if (file.exists("renv.lock")) {
  # ... restore it 
  renv::restore(lockfile = "renv.lock", prompt = FALSE, clean = TRUE)  
} else {
  # ... otherwise, install from list of required packages
  pkgs <- readLines("replication_r_requirements.txt")
  # TODO: change to "r_requirements.txt" when wanting to run all R code in code/0[1-4]-*/
  
  res <- character()
  for (pkg in pkgs) {
    message("installing ", pkg)
    # try install
    tmp <- tryCatch(renv::install(pkg, prompt = FALSE, lock = TRUE), error = function(err) err)
    if (inherits(tmp, "error")) {
      # try install binary
      tmp <- tryCatch(renv::install(pkg, type = "binary", prompt = FALSE, lock = TRUE), error = function(err) err)
      if (inherits(tmp, "error")) {
        res[pkg] <- tmp$message
        next
      }
    }
    res[pkg] <- "success"
  }
  if (any(idxs <- res!="success")) {
    issues <- res[idxs]
    warning("Could not install the following packages with given versions: ", paste(names(issues), collapse = ", "), call. = FALSE, immediate. = TRUE)
    cat(paste("\n+++ ISSUE +++", names(issues), "", issues, sep = "\n"), sep = "\n\n==================================\n")
  }
  
}
