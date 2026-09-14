"""Reusable PDF: scoringutils summary plus matched rolling forecast panels."""
from __future__ import annotations

import argparse
import io
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from influpaintx.data.geography import STATE_FIPS
from .hubs import KEY

TITLES = {'wk inc flu hosp': 'Influenza admissions', 'wk inc flu prop ed visits': 'Influenza ED proportion',
          'wk inc covid hosp': 'COVID-19 admissions', 'wk inc covid prop ed visits': 'COVID-19 ED proportion',
          'wk inc rsv hosp': 'RSV admissions', 'wk inc rsv prop ed visits': 'RSV ED proportion'}
COLORS = ['#127b83', '#c17022', '#7657a5']


def nice(value, target):
    return f'{value:.5f}' if 'prop ed' in target else f'{value:.2f}'


def cover(manifest, path, locations):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=12, leading=17, spaceAfter=10))
    styles.add(ParagraphStyle(name='SmallReport', fontName='Helvetica', fontSize=10, leading=13, spaceAfter=6))
    body, small = styles['BodyReport'], styles['SmallReport']
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A3), rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=38)
    story = [Paragraph('InfluPaintX B0 | Hub forecast comparisons', styles['Title']), Spacer(1, 12),
             Paragraph('Pinned local archives - R scoringutils - finalized-data retrospective cross-validation', body),
             Paragraph('All available ensemble-supported units are scored. This report plots ' + ', '.join(locations) +
                       ' with all four horizons, using only the median, 50% and 95% intervals. The model, best eligible submitted competitor, and official ensemble have separate columns with matching axes.', body)]
    data = [[Paragraph(v, small) for v in ['Target / season', 'Best eligible submitted competitor', 'B0 WIS', 'Best WIS', 'Ensemble WIS', 'B0 50% / 95% coverage', 'Scored units']]]
    for case in manifest['cases']:
        if case['status'] != 'scored':
            continue
        rows = {r['model']: r for r in case['summary']}
        ours, ensemble = rows[case['model_name']], rows[case['ensemble']]
        best = rows.get(case['best'])
        values = [TITLES[case['target']] + '<br/>' + case['season'], case['best'] or 'No complete eligible competitor',
                  nice(ours['wis'], case['target']), nice(best['wis'], case['target']) if best else '-',
                  nice(ensemble['wis'], case['target']),
                  f"{100 * ours['interval_coverage_50']:.1f}% / {100 * ours['interval_coverage_95']:.1f}%", f"{case.get('n_comparison_units', case['n_units']):,}" + ('*' if case.get('best_coverage_fallback') else '')]
        data.append([Paragraph(str(v), small) for v in values])
    table = Table(data, colWidths=[185, 220, 92, 92, 102, 165, 80], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5eff1')),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                               ('TOPPADDING', (0, 0), (-1, -1), 6),
                               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f7f8')])]))
    story += [table, Spacer(1, 12),
              Paragraph('WIS is lower-is-better and is not comparable across counts and proportions. Coverage fractions from scoringutils are displayed as percentages; nominal levels are 50% and 95%. Scores in each row pool identical comparison units, including US and hub horizons 0-3. An asterisk marks a reduced comparison set when no competitor has complete coverage; the full ensemble-supported scores are retained in the CSVs.', small),
              Paragraph('"Best" is the lowest mean WIS among submitted competitors covering 100% of this exact task set; official baseline and official ensemble are excluded from winning. If no complete competitor exists, submissions with at least 90% coverage are ranked on their identical shared units; all three displayed models use that same reduced set. Other partial submissions are scored and listed separately. This is not an official hub leaderboard or a claim about all possible ranking methods.', small),
              PageBreak(), Paragraph('How to interpret these comparisons', styles['Heading1'])]
    for title, text in [
        ('Information advantage', 'B0 uses finalized NHSN and a frozen latest NSSP snapshot. Hub models were submitted prospectively with provisional information. Train 2024-25 + 2025-26 -> evaluate 2023-24, and train 2023-24 + 2025-26 -> evaluate 2024-25, use later seasons for fitting. Only train 2023-24 + 2024-25 -> evaluate 2025-26 is chronological; even it is a finalized-input comparison. None of these are operational leaderboard results.'),
        ('Exact timing', 'Our 1-4 weeks after context end become hub horizons 0-3 with reference Saturday = context end + 7 days. Every target_end_date is checked against reference_date + 7*horizon. All forecasts are saved cross-validation forecasts; no fitting or tuning was performed for this comparison.'),
        ('Scoring universe', 'Each target/season uses only complete, valid ensemble quantile tasks that overlap the held-out B0 forecasts and have observed hub truth. We do not add unsubmitted locations, off-ensemble dates, or targets without an ensemble. The latest full target-data release at each pinned commit is used for every model; omissions are not filled from earlier releases.'),
        ('Quantiles and scores', 'All 23 hub quantile levels, extracted from 2,048 B0 draws, are scored with R scoringutils::as_forecast_quantile and scoringutils::score. The metrics and subprocess pattern follow ../epibench/src/epibench/scoring_bridge.py. B0 admission quantiles retain their existing half-up rounding. Hub predictions remain as submitted. WIS, absolute error of the median, 50/95 coverage, dispersion, underprediction, overprediction, and bias are retained.'),
        ('Target compatibility', 'Only the current CDC RSV hub is used. No older RSV-NET or catchment targets are scored. COVID means the Forecast Hub, as clarified by the user; no scenario projections are used. FluSight ED, RSV ED, and COVID ED are proportions in [0,1], not percentages.'),
        ('Figures', 'Rows are forecast leads 1-4 weeks after context end (hub horizons 0-3); columns are B0, best eligible submitted competitor, and ensemble. Shaded ribbons show 95% (light) and 50% (dark) prediction intervals, a colored line shows the median, and a black line shows the same frozen observed truth. Curves are rolling forecasts from different reference dates, not a single season trajectory. Every column uses the same dates for the displayed location/horizon. Lines break across missing weekly dates. The selected locations are illustrative, not selected for forecast performance.'),
    ]:
        story += [Paragraph(title, styles['Heading3']), Paragraph(text, body)]
    story += [Paragraph('Pinned sources and coverage', styles['Heading2'])]
    for hub, info in manifest['hubs'].items():
        versions = '; '.join(f'{TITLES[t]}: {v}' for t, v in info['truth_vintages'].items())
        story.append(Paragraph(f"{hub}: {info['url']}<br/>commit {info['commit']}<br/>Frozen truth releases: {versions}", small))
    unavailable = [f"{TITLES[c['target']]} {c['season']}" for c in manifest['cases'] if c['status'] != 'scored']
    story.append(Paragraph('No comparable ensemble-scored units: ' + '; '.join(unavailable) + '.', small))
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#58646a'))
        canvas.drawString(42, 20, 'InfluPaintX research pilot | matched tasks, finalized data | full scores and provenance accompany this PDF')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def plot_pages(manifest, folder, path, requested):
    count = 0
    with PdfPages(path) as pdf:
        for case in manifest['cases']:
            if case['status'] != 'scored':
                continue
            wide = pd.read_parquet(folder / case['directory'] / 'quantiles.parquet')
            models = [case['model_name'], case['best'], case['ensemble']]
            wide = wide[wide.model.isin([m for m in models if m])]
            unit_file = folder / case['directory'] / 'comparison_units.parquet'
            if unit_file.exists():
                wide = wide.merge(pd.read_parquet(unit_file), on=KEY, validate='many_to_one')
            available = sorted(wide.location.unique())
            mapping = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
            locations = available if requested == ['all'] else [mapping.get(v, v) for v in requested]
            for loc in locations:
                local = wide[wide.location == loc]
                if local.empty:
                    continue
                fig, axes = plt.subplots(4, 3, figsize=(16.54, 11.69), sharex=True,
                                         gridspec_kw={'hspace': .31, 'wspace': .12})
                fig.subplots_adjust(left=.065, right=.98, bottom=.07, top=.845)
                display = STATE_FIPS.get(loc, loc)
                title = f"{TITLES[case['target']]} | {case['season']} | {display}"
                fig.suptitle(title, x=.065, y=.974, ha='left', fontsize=20, weight='bold')
                fig.text(.065, .943, f"{case['n_reference_dates']} ensemble reference dates in full scored panel | matched hub horizons 0-3 | finalized-input CV" + (' | >=90% coverage comparison subset' if case.get('best_coverage_fallback') else ''), fontsize=11, color='#4c5c61')
                handles = [Line2D([0], [0], color='#555555', label='Median'),
                           Patch(facecolor='#8a9a9f', alpha=.42, label='50% interval'),
                           Patch(facecolor='#8a9a9f', alpha=.18, label='95% interval'),
                           Line2D([0], [0], color='black', label='Observed')]
                fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .93), ncol=4, frameon=False, fontsize=10)
                for h in range(4):
                    horizon = local[local.horizon == h]
                    ymax = max(horizon['q0.975'].max(), horizon.observed.max())
                    for j, model in enumerate(models):
                        ax = axes[h, j]
                        if model is None:
                            ax.text(.5, .5, 'No complete eligible competitor', ha='center', va='center', transform=ax.transAxes)
                            continue
                        data = horizon[horizon.model == model].sort_values('target_end_date')
                        if len(data):
                            dates = pd.to_datetime(data.target_end_date)
                            full = pd.date_range(dates.min(), dates.max(), freq='7D')
                            data = data.assign(target_end_date=dates).set_index('target_end_date').reindex(full)
                            ax.fill_between(full, data['q0.025'].to_numpy(), data['q0.975'].to_numpy(), color=COLORS[j], alpha=.18, linewidth=0)
                            ax.fill_between(full, data['q0.25'].to_numpy(), data['q0.75'].to_numpy(), color=COLORS[j], alpha=.38, linewidth=0)
                            ax.plot(full, data['q0.5'], color=COLORS[j], linewidth=1.4)
                            ax.plot(full, data.observed, color='#171717', linewidth=1.0)
                        ax.set_ylim(0, ymax * 1.06 if ymax > 0 else 1)
                        ax.grid(axis='y', alpha=.18)
                        ax.spines[['top', 'right']].set_visible(False)
                        ax.tick_params(labelsize=8)
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                        if j == 0:
                            ax.set_ylabel(f"Lead {h+1} week(s) / hub h={h}\n" + ('Proportion' if 'prop ed' in case['target'] else 'Admissions'), fontsize=10)
                        else:
                            ax.tick_params(labelleft=False)
                        if h == 0:
                            short = ['B0 finalized CV', model, case['ensemble']][j]
                            ax.set_title(short, fontsize=11, color=COLORS[j], pad=12)
                        if h == 3:
                            ax.set_xlabel('Target week', fontsize=10)
                fig.text(.065, .024, 'Same observed truth and ensemble-scored units in all columns. Best = minimum mean WIS on identical supported tasks (coverage rule on page 1); not an official rank.', fontsize=9, color='#4c5c61')
                pdf.savefig(fig)
                plt.close(fig)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=Path('data/evaluation/b0_hub_comparison'))
    parser.add_argument('--output', type=Path, default=Path('output/pdf/b0_hub_comparison.pdf'))
    parser.add_argument('--locations', nargs='+', default=['US', 'NC'])
    args = parser.parse_args()
    manifest = json.loads((args.comparison / 'manifest.json').read_text())
    tmp = Path('tmp/pdfs')
    tmp.mkdir(parents=True, exist_ok=True)
    cover(manifest, tmp / 'comparison-cover.pdf', args.locations)
    plots = plot_pages(manifest, args.comparison, tmp / 'comparison-plots.pdf', args.locations)
    writer = PdfWriter()
    for path in [tmp / 'comparison-cover.pdf', tmp / 'comparison-plots.pdf']:
        writer.append(path)
    for number, page in enumerate(writer.pages, 1):
        stamp = io.BytesIO()
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        canvas = Canvas(stamp, pagesize=(width, height))
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#58646a'))
        canvas.drawRightString(width - 42, 20, f'{number} / {len(writer.pages)}')
        canvas.save()
        stamp.seek(0)
        page.merge_page(PdfReader(stamp).pages[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        writer.write(stream)
    print(json.dumps({'pdf': str(args.output), 'pages': len(PdfReader(args.output).pages), 'plot_pages': plots}))


if __name__ == '__main__':
    main()
