# Season peak by autoregressive rollout of A (7 October 2026)

No fitting. Saved checkpoints of recipe A (width-96 sampled MLP with Kinsa, histories
degraded then corrected by real-pair trees; finalized flu admissions/ED labels) are run
forward: each sampled four-week forecast is appended to the history as finalized flu
admissions and ED, the 12-week window shifts four weeks, repeat to end of May. Covid/RSV
and Kinsa are marked unavailable in simulated weeks. A was trained only on one-step
four-week forecasts. 5 seeds x 200 paths. Peak = max weekly admissions from the first
Saturday of October to the last Saturday of May (observed corrected reports + simulated
weeks); peak week = argmax. Scored against the finalized frozen panel
(`data/processed/panel.npz`). Code: `scripts/peak_rollout.py`, `scripts/peak_report.py`.

Evaluations (each A seed 44-48 from `b6-search-full-20261007`):
- 2025-26 forecast by A trained on 2022-23, 2023-24, 2024-25 (forward check).
- 2024-25 forecast by A trained on 2022-23, 2023-24, 2025-26 (uses a later season).

Columns (scores CSVs): share of 52 locations whose true peak lies in the 50%/95% peak
interval (target 0.5/0.95), median over locations of forecast median / true peak
(1 is best), mean probability on the true peak week exactly and within +-1 week (higher is better).

| Forecast date | Season | 95% coverage | Median / true | P(week +-1) | US median peak (true) | US modal week (true) |
|---|---|---:|---:|---:|---|---|
| 8-9 Oct | 2025-26 | 0.92 | 0.89 | 0.16 | 54,900 (42,510) | 21 Feb (3 Jan) |
| 3-4 Dec | 2025-26 | 0.96 | 0.90 | 0.27 | 47,400 (42,510) | 24 Jan (3 Jan) |
| 9 Oct | 2024-25 | 0.50 | 0.37 | 0.23 | 29,000 (55,551) | 25 Jan (8 Feb) |
| 4 Dec | 2024-25 | 0.87 | 0.70 | 0.34 | 43,400 (55,551) | 25 Jan (8 Feb) |
| 15 Jan | 2024-25 | 0.17 | 0.65 | 0.28 | 39,700 (55,551) | 4 Jan (8 Feb) |

Findings: 2025-26 reasonably calibrated for size but peak placed 3-7 weeks late.
2024-25 (the larger, later season) peak size strongly underestimated and placed too early,
with poor coverage through January. Timing errors go both ways, so "a bit late" is not a
fixable constant shift. After the peak, intervals collapse on the observed reported peak
and miss the finalized value (coverage ~0.1-0.4): observed season weeks should use their
correction/uncertainty, not a single reported value. Not submitted.
