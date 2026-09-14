#!/usr/bin/env Rscript
# R itself must already be installed. Reuse available packages; install missing ones.
packages <- c("scoringutils", "purrr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  user_library <- strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep, fixed = TRUE)[[1]]
  if (!length(user_library) || !nzchar(user_library[1])) {
    stop("Set R_LIBS_USER to a writable personal R library, then rerun this script.")
  }
  user_library <- path.expand(user_library[1])
  dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(user_library, .libPaths()))
  install.packages(missing, lib = user_library, repos = "https://cloud.r-project.org")
}
for (package in packages) {
  library(package, character.only = TRUE)
  cat(package, as.character(packageVersion(package)), "\n")
}
cat(R.version.string, "\n")
