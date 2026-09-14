# Solution B reference archive

Prepared 5 September 2026 for the [detailed plan](../docs/design/solution-b-plan.md).

16 PDFs downloaded and validated with `pdfinfo`; the Scheuerer–Hamill PDF was read through the web tool but its local download returned HTTP 403. Its official source link remains available below. Preprints and published versions may differ. File hashes and detected arXiv versions are recorded in [manifest.json](manifest.json). Papers retain their original licenses.

Downloaded PDFs, extracted text, and local data audit files are excluded from Git.
Local PDF links below work after running the downloader; the source links work
without downloading. The bibliography, download script, and paper manifest are included.

Reading order: Flusion, FGN, Pacchiardi, Ferro, Bracher, CSDI, Scheuerer–Hamill, InfluPaint, followed by architecture and predecessor context.

| Paper | Role | Local PDF / source | Archived version |
|---|---|---|---|
| Flusion: Integrating multiple data sources for accurate influenza predictions — Evan L. Ray; Yijin Wang; Russell D. Wolfinger; Nicholas G. Reich (2024) | Core: transfer across signals, locations, transforms, and baseline | [PDF](ray2024_flusion.pdf) · [source](https://arxiv.org/abs/2407.19054v1) | 2407.19054v1 |
| Generative diffusion models for spatiotemporal influenza forecasting — Joseph Lemaitre; Justin Lessler (2026) | Core: simulation mixing experiment and calibration diagnostics | [PDF](lemaitre2026_influpaint.pdf) · [source](https://arxiv.org/abs/2604.24913v1) | 2604.24913v1 |
| Skillful joint probabilistic weather forecasting from marginals — Ferran Alet and colleagues (2025) | Core: global functional noise and fair CRPS | [PDF](alet2025_fgn.pdf) · [source](https://arxiv.org/abs/2506.10772v1) | 2506.10772v1 |
| GenCast: Diffusion-based ensemble forecasting for medium-range weather — Ilan Price and colleagues (2023) | Context: conditional transitions and optional rollout | [PDF](price2023_gencast.pdf) · [source](https://arxiv.org/abs/2312.15796v2) | 2312.15796v2 |
| CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation — Yusuke Tashiro; Jiaming Song; Yang Song; Stefano Ermon (2021) | Core: separate conditioning and target masks | [PDF](tashiro2021_csdi.pdf) · [source](https://arxiv.org/abs/2107.03502v2) | 2107.03502v2 |
| FiLM: Visual Reasoning with a General Conditioning Layer — Ethan Perez; Florian Strub; Harm de Vries; Vincent Dumoulin; Aaron Courville (2018) | Mechanism: feature-wise affine modulation | [PDF](perez2018_film.pdf) · [source](https://arxiv.org/abs/1709.07871v2) | 1709.07871v2 |
| TSMixer: An All-MLP Architecture for Time Series Forecasting — Si-An Chen; Chun-Liang Li; Nate Yoder; Sercan O. Arik; Tomas Pfister (2023) | Architecture alternative: temporal and feature mixing | [PDF](chen2023_tsmixer.pdf) · [source](https://arxiv.org/abs/2303.06053v5) | 2303.06053v5 |
| Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles — Balaji Lakshminarayanan; Alexander Pritzel; Charles Blundell (2017) | Core: independently initialized model ensemble | [PDF](lakshminarayanan2017_deep_ensembles.pdf) · [source](https://arxiv.org/abs/1612.01474v3) | 1612.01474v3 |
| Fair scores for ensemble forecasts — Christopher A. T. Ferro (2014) | Scoring: finite-ensemble correction; author manuscript | [PDF](ferro2014_fair_scores.pdf) · [source](https://doi.org/10.1002/qj.2270) | Publisher/author PDF |
| Evaluating epidemic forecasts in an interval format — Johannes Bracher; Evan L. Ray; Tilmann Gneiting; Nicholas G. Reich (2021) | Scoring: WIS, quantile loss and calibration | [PDF](bracher2021_interval_scores.pdf) · [source](https://arxiv.org/abs/2005.12881v3) | 2005.12881v3 |
| Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities — Michael Scheuerer; Thomas M. Hamill (2015) | Scoring: dependence-sensitive diagnostics | [source](https://doi.org/10.1175/MWR-D-14-00269.1) | Not downloaded |
| Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization — Lorenzo Pacchiardi; Rilwan Adewoyin; Peter Dueben; Ritabrata Dutta (2024) | Core precedent: conditional implicit generators trained by proper scores | [PDF](pacchiardi2024_scoring_rules.pdf) · [source](https://jmlr.org/papers/v25/23-0038.html) | Publisher/author PDF |
| Flow Matching for Generative Modeling — Yaron Lipman; Ricky T. Q. Chen; Heli Ben-Hamu; Maximilian Nickel; Matt Le (2023) | Comparator only: alternate conditional generative objective | [PDF](lipman2023_flow_matching.pdf) · [source](https://arxiv.org/abs/2210.02747v2) | 2210.02747v2 |
| RePaint: Inpainting using Denoising Diffusion Probabilistic Models — Andreas Lugmayr and colleagues (2022) | Background only: conditioning at sampling in predecessor | [PDF](lugmayr2022_repaint.pdf) · [source](https://arxiv.org/abs/2201.09865v4) | 2201.09865v4 |
| Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models — Guanhua Zhang and colleagues (2023) | Background only: predecessor inpainting | [PDF](zhang2023_copaint.pdf) · [source](https://arxiv.org/abs/2304.03322v1) | 2304.03322v1 |
| Denoising Diffusion Probabilistic Models — Jonathan Ho; Ajay Jain; Pieter Abbeel (2020) | Background only: predecessor diffusion objective | [PDF](ho2020_ddpm.pdf) · [source](https://arxiv.org/abs/2006.11239v2) | 2006.11239v2 |
| Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow — Xingchao Liu; Chengyue Gong; Qiang Liu (2023) | Comparator only: straight interpolation transport | [PDF](liu2023_rectified_flow.pdf) · [source](https://arxiv.org/abs/2209.03003v1) | 2209.03003v1 |

Additional evidence: [FluSight README snapshot](flusight-hub-readme-2026-09-05.md), [local latest-view data inventory](local-data-coverage-2026-09-05.json), and [BibTeX](references.bib). The local inventory is not a historical as-of eligibility audit.

Re-run downloads with `python3 references/download_references.py` from the project root. Existing valid PDFs are retained. `text/` contains extracted PDF text for local search. A failed download is recorded, not saved as a fake PDF.
