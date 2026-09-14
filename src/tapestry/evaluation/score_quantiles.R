#!/usr/bin/env Rscript
# Same external-R bridge pattern and metrics as ../epibench/scoring_bridge.py.
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(scoringutils))
suppressPackageStartupMessages(library(purrr))
df <- read.csv(args[1], stringsAsFactors = FALSE,
               colClasses = c(location = "character"))
df$target_end_date <- as.Date(df$target_end_date)
forecast_object <- as_forecast_quantile(
  df, forecast_unit = c("model", "reference_date", "target_end_date", "location", "horizon"),
  observed = "observed", predicted = "predicted", quantile = "quantile_level"
)
metrics <- list(
  wis = wis, overprediction = overprediction_quantile,
  underprediction = underprediction_quantile, dispersion = dispersion_quantile,
  bias = bias_quantile, interval_coverage_50 = interval_coverage,
  interval_coverage_95 = partial(interval_coverage, interval_range = 95),
  ae_median = ae_median_quantile
)
scores <- score(forecast_object, metrics = metrics)
write.csv(scores, args[2], row.names = FALSE)
writeLines(c(paste("R", getRversion()), paste("scoringutils", packageVersion("scoringutils")),
             paste("purrr", packageVersion("purrr"))), args[3])
