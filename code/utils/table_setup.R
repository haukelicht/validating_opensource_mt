require(readr, quietly = TRUE, warn.conflicts = FALSE)
require(purrr, quietly = TRUE, warn.conflicts = FALSE)
require(stringr, quietly = TRUE, warn.conflicts = FALSE)
require(knitr, quietly = TRUE, warn.conflicts = FALSE)
suppressWarnings(require(kableExtra, quietly = TRUE, warn.conflicts = FALSE))

options(knitr.table.format = "latex")
options(knitr.kable.NA = "")
options(knitr.kable.digits = 3)

dflt_kable_style <- partial(kable_styling, full_width = FALSE, position = "center", font_size = 10L)
use_kable <- partial(kable, format = "latex", digits = 3, booktabs = TRUE, linesep = "")
quick_kable <- function(...) dflt_kable_style(use_kable(...))

save_kable <- function(
    x,
    dir,
    overwrite = FALSE,
    replace.column.types = NULL,
    .file.name = sub(".*\\\\label\\{([^{]+)\\}.*", "\\1", attr(x, "kable_meta")$caption, perl = TRUE),
    .file.name.cleaning = c("^tab:" = ""),
    .file.extension = "tex",
    .write = TRUE,
    .position = "!t",
    .return.string = FALSE
) {
  if (!is.null(.file.name.cleaning))
    .file.name <- stringr::str_replace_all(.file.name, .file.name.cleaning)
  
  fp <- file.path(dir, paste0(.file.name, ".", .file.extension))
  if (!is.null(.position))
    x <- sub("\\begin{table}\n", sprintf("\\begin{table}[%s]\n", .position), as.character(x), fixed = TRUE)

  if (!is.null(replace.column.types)) {
    names(replace.column.types) <- paste0("(?<=\\b)", names(replace.column.types), "(?=\\{(\\d|\\*))")
    x <- stringr::str_replace_all(as.character(x), replace.column.types)
  }
  
  if (!file.exists(fp) || overwrite)
    readr::write_lines(as.character(x), fp)
  if (.return.string)
    return(x)
}


escape_latex <- function(x) {
  pats <- c(
    "&" = "\\\\&",
    '(?<=\\s)"(?=\\S)' = "``",
    NULL
  )
  str_replace_all(x, pats)
}
