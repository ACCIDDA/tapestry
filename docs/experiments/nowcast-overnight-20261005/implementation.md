# Private implementation and integration

The research ran from `/tmp/chromantis-nowcast-overnight-20261005` locally and
`/proj/jlessler/projects/tapestry-all/tapestry-nowcast-overnight-20261005` remotely.
Shared source was not overwritten. The parent already integrated the separate
joint-loss wiring correction from `joint-loss-wiring.patch`.

`nowcast-mechanisms-complete.patch` contains subsequent core changes against the
initial overnight pinned source, which already included that loss correction.
It supersedes the earlier partial `nowcast-mechanisms.patch` for core integration.
Apply by review because the shared workspace has simultaneous vintage-model work;
do not replace whole shared files with the private copies. `research-source.tar.gz`
contains the current private changed files and research scripts for reproduction.
It is a source snapshot, not an instruction to overwrite shared work.

Core changes:

- `model/nowcast_overnight.py`: bounded nonlinear log-revision correction from
  reported trajectory, availability, calendar/location and optional covariates.
- `experiment/nowcast_overnight.py`: aligned fixed-forecaster input replay,
  strict source provenance, native-unit diagnostics, coherent history mixtures,
  and season-cross-fitted synthetic residual uncertainty.
- `experiment/weekend.py`: nonlinear corrector selection, optional observed-pair
  supervision filter, fixed-forecaster replay routing and input diagnostics.
- `model/scenario.py`: explicit private scenario fields for these mechanisms;
  restores separate meaning of model dropout and reporting availability.
- `model/network.py`: optional stochastic current-level anchoring of future
  decoder residuals. Its pilots were unsuccessful; default behavior is unchanged.
- `dataset/reporting_error.py`: donor archived-pair support metadata, distinct
  from final-label support. It does not change the generated values or RNG.
- `experiment/dispatch.py`: reads `DEVICE=cpu` for CPU inference jobs; default cuda.

Two small non-core integration changes are in the source archive rather than the
core patch: add `scikit-learn>=1.5` to model dependencies (cluster version1.7.2 was
already installed), and skip the CUDA assertion in `scripts/jlessler.sbatch` when
`DEVICE=cpu`. The notification wrapper is pinned to node2 as well. The CPU jobs also override `--gres=gpu:0`; they used node2 only.

Every experiment retains its own immutable `code/` snapshot. Historical pilots
and later mechanisms did not execute from a changing live source tree. The initial
26-run allocation and trajectory21-run allocation are documented explicitly;
research scripts should be run with those allocations, not a full architecture
or three-seed sweep of all deferred pilots.

The fixed replay supports admission-only correction and removing nowcaster covariates. Neither changes the saved forecaster. The principal export script preserves matched hardware/draw controls and both seasonal definitions.

The final source snapshot also includes the C1 full/half retrained-pipeline planner,
its exact-corrector H100 replay planner/report, the H1002048 mixture confirmation,
and a device-aware completed-run inventory. The final CPU plotting script filters
CPU comparisons explicitly so the newly added H100 rows cannot enter its figure.
All final source/figure artifacts were refreshed after the07:13 follow-up cohort.
