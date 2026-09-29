# Cleanup record

Removed retired B2 experiments, plans, derived panels, reports and scripts locally
and on Longleaf. Shared raw source downloads and the current panel were retained.

The local deletion inventory is `removed-legacy.json`. A substring match also
removed the regenerable legacy `nowcast-pipeline-smoke/ranking-a93fb28a6073`
cache because its hash contained `b2`; it was not a B2 experiment. Subsequent
cleanup uses experiment-name prefixes only. No raw source data was removed.
