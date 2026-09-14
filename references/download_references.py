"""Download the papers used by the Solution B plan; retain provenance and checksums.

Run with Python 3.11+: python references/download_references.py
Existing valid PDFs are retained. This script does not redistribute or relicense papers.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent


def arxiv(key, title, authors, year, identifier, role):
    return dict(key=key, title=title, authors=authors, year=year,
                url=f"https://arxiv.org/abs/{identifier}",
                pdf_url=f"https://arxiv.org/pdf/{identifier}", role=role)


PAPERS = [
    arxiv("ray2024_flusion", "Flusion: Integrating multiple data sources for accurate influenza predictions", "Evan L. Ray; Yijin Wang; Russell D. Wolfinger; Nicholas G. Reich", 2024, "2407.19054v1", "Core: transfer across signals, locations, transforms, and baseline"),
    arxiv("lemaitre2026_influpaint", "Generative diffusion models for spatiotemporal influenza forecasting", "Joseph Lemaitre; Justin Lessler", 2026, "2604.24913v1", "Core: simulation mixing experiment and calibration diagnostics"),
    arxiv("alet2025_fgn", "Skillful joint probabilistic weather forecasting from marginals", "Ferran Alet and colleagues", 2025, "2506.10772v1", "Core: global functional noise and fair CRPS"),
    arxiv("price2023_gencast", "GenCast: Diffusion-based ensemble forecasting for medium-range weather", "Ilan Price and colleagues", 2023, "2312.15796", "Context: conditional transitions and optional rollout"),
    arxiv("tashiro2021_csdi", "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", "Yusuke Tashiro; Jiaming Song; Yang Song; Stefano Ermon", 2021, "2107.03502", "Core: separate conditioning and target masks"),
    arxiv("perez2018_film", "FiLM: Visual Reasoning with a General Conditioning Layer", "Ethan Perez; Florian Strub; Harm de Vries; Vincent Dumoulin; Aaron Courville", 2018, "1709.07871", "Mechanism: feature-wise affine modulation"),
    arxiv("chen2023_tsmixer", "TSMixer: An All-MLP Architecture for Time Series Forecasting", "Si-An Chen; Chun-Liang Li; Nate Yoder; Sercan O. Arik; Tomas Pfister", 2023, "2303.06053", "Architecture alternative: temporal and feature mixing"),
    arxiv("lakshminarayanan2017_deep_ensembles", "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", "Balaji Lakshminarayanan; Alexander Pritzel; Charles Blundell", 2017, "1612.01474", "Core: independently initialized model ensemble"),
    dict(key="ferro2014_fair_scores", title="Fair scores for ensemble forecasts", authors="Christopher A. T. Ferro", year=2014, url="https://doi.org/10.1002/qj.2270", pdf_url="https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf", role="Scoring: finite-ensemble correction; author manuscript"),
    arxiv("bracher2021_interval_scores", "Evaluating epidemic forecasts in an interval format", "Johannes Bracher; Evan L. Ray; Tilmann Gneiting; Nicholas G. Reich", 2021, "2005.12881", "Scoring: WIS, quantile loss and calibration"),
    dict(key="scheuerer2015_variogram", title="Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities", authors="Michael Scheuerer; Thomas M. Hamill", year=2015, url="https://doi.org/10.1175/MWR-D-14-00269.1", pdf_url="https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf", role="Scoring: dependence-sensitive diagnostics"),
    dict(key="pacchiardi2024_scoring_rules", title="Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization", authors="Lorenzo Pacchiardi; Rilwan Adewoyin; Peter Dueben; Ritabrata Dutta", year=2024, url="https://jmlr.org/papers/v25/23-0038.html", pdf_url="https://jmlr.org/papers/volume25/23-0038/23-0038.pdf", role="Core precedent: conditional implicit generators trained by proper scores"),
    arxiv("lipman2023_flow_matching", "Flow Matching for Generative Modeling", "Yaron Lipman; Ricky T. Q. Chen; Heli Ben-Hamu; Maximilian Nickel; Matt Le", 2023, "2210.02747", "Comparator only: alternate conditional generative objective"),
    arxiv("lugmayr2022_repaint", "RePaint: Inpainting using Denoising Diffusion Probabilistic Models", "Andreas Lugmayr and colleagues", 2022, "2201.09865", "Background only: conditioning at sampling in predecessor"),
    arxiv("zhang2023_copaint", "Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models", "Guanhua Zhang and colleagues", 2023, "2304.03322", "Background only: predecessor inpainting"),
    arxiv("ho2020_ddpm", "Denoising Diffusion Probabilistic Models", "Jonathan Ho; Ajay Jain; Pieter Abbeel", 2020, "2006.11239", "Background only: predecessor diffusion objective"),
    arxiv("liu2023_rectified_flow", "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", "Xingchao Liu; Chengyue Gong; Qiang Liu", 2023, "2209.03003", "Comparator only: straight interpolation transport"),
]


def download(paper):
    record = dict(paper)
    path = ROOT / f"{paper['key']}.pdf"
    try:
        if not path.exists() or not path.read_bytes().startswith(b"%PDF-"):
            request = urllib.request.Request(paper["pdf_url"], headers={"User-Agent": "InfluPaintX-reference-archive/1.0"})
            with urllib.request.urlopen(request, timeout=50) as response:
                payload = response.read()
                record["resolved_url"] = response.url
            if not payload.startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF")
            path.write_bytes(payload)
        payload = path.read_bytes()
        record.update(file=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), status="downloaded")
        text_path = ROOT / "text" / f"{paper['key']}.txt"
        text_path.parent.mkdir(exist_ok=True)
        try:
            subprocess.run(["pdftotext", "-layout", str(path), str(text_path)], check=True, capture_output=True)
            record["text_file"] = str(text_path.relative_to(ROOT))
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            record["extraction_note"] = str(exc)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
    record["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(record["key"], record["status"], record.get("bytes", record.get("error")), flush=True)
    return record


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, PAPERS))
    (ROOT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    if any(record["status"] != "downloaded" for record in records):
        raise SystemExit("Some downloads failed; see references/manifest.json")
