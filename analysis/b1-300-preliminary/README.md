# Preliminary B1 300-epoch comparison

Snapshot: 41/48 completed runs. Only complete runs enter this analysis; comparisons use identical architecture/formulation/seed pairs, all with mixed 50% masking. Missing seeds are not imputed.

Forecast relative WIS; lower is better, 1 is Hub ensemble parity. The scientific weighting and frozen inputs match the original screen (input hashes, score version, and scored support checked). This is retrospective development CV, not prospective performance. Caps are maxima with early stopping, not fixed training lengths.

| Configuration | Seeds | Original screen, matched seeds | Cap 300 | Change | Seeds improved |
|---|---:|---:|---:|---:|---:|
| joint_mlp__direct__mask0.5 | 3/3 | 1.012 | 1.086 | +0.074 | 1 |
| joint_mlp__direct_finalflag__mask0.5 | 3/3 | 0.990 | 0.991 | +0.000 | 0 |
| joint_mlp__joint_aux025__mask0.5 | 3/3 | 0.972 | 0.983 | +0.011 | 1 |
| joint_mlp__two_stage__mask0.5 | 3/3 | 1.211 | 1.194 | -0.017 | 2 |
| pathogen_mlp__direct__mask0.5 | 3/3 | 1.045 | 1.126 | +0.081 | 1 |
| pathogen_mlp__direct_finalflag__mask0.5 | 3/3 | 0.952 | 0.941 | -0.011 | 3 |
| pathogen_mlp__joint_aux025__mask0.5 | 3/3 | 0.957 | 0.945 | -0.011 | 2 |
| pathogen_mlp__two_stage__mask0.5 | 3/3 | 1.104 | 1.292 | +0.189 | 1 |
| target_mlp__direct__mask0.5 | 1/3 | 1.040 | 1.042 | +0.002 | 0 |
| target_mlp__direct_finalflag__mask0.5 | 2/3 | 0.999 | 1.066 | +0.067 | 0 |
| target_mlp__joint_aux025__mask0.5 | 3/3 | 0.985 | 1.003 | +0.018 | 0 |
| target_mlp__two_stage__mask0.5 | 3/3 | 1.179 | 1.203 | +0.024 | 1 |
| target_multiscale__direct__mask0.5 | 2/3 | 1.116 | 1.072 | -0.044 | 1 |
| target_multiscale__direct_finalflag__mask0.5 | 3/3 | 1.031 | 1.032 | +0.001 | 2 |
| target_multiscale__two_stage__mask0.5 | 3/3 | 1.206 | 1.215 | +0.009 | 2 |

## Interpretation

- Pathogen B improves from 0.952 to 0.941, in all three seeds; pathogen C improves from 0.957 to 0.945, in two seeds. The B/C difference remains small compared with seed variability.
- All four two-stage configurations have three seeds complete and remain worse than the ensemble. Pathogen two-stage worsens from 1.104 to 1.292; much of the change is seed 42 (1.082 to 1.606). Longer training has not rescued this formulation.
- Target C worsens in all three matched seeds, from 0.985 to 1.003. Target B is incomplete (2/3), with both available seeds worse.
- Joint MLP already had cap 300. Its rows are repeat controls, not duration effects. Joint direct seed 44 changes from 1.144 to 1.363 despite the same cap and seed; do not attribute all observed differences to duration. Cause of this repeat variability is not established here. Source differences are limited to suite selection/planning, not fitting code.
- The original gap-only winner is outside this mixed-mask follow-up. Comparing 0.941 here with its 0.939 does not isolate a training-budget effect.
- No standalone nowcast ranking is included. Forecast skill does not establish nowcast accuracy.

## Log

- 2026-09-17: aggregated saved scores for the 41 completed runs and compared matched seeds against the first screen. No training or scoring jobs launched. The online first-screen report remains a separate experiment snapshot.
