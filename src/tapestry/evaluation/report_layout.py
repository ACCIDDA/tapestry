"""One report layout, applied to saved text and figures without recomputing plots."""
import base64
import hashlib
import html
import json
import re
from pathlib import Path

SECTIONS = ('Best models by season', 'Standard evaluation', 'Protocol', 'Season splits', 'Findings',
            'Forecast fans', 'Score diagnostics', 'Matched comparisons', 'Full ranking', 'Appendix')
START, END = '<!-- write-up: kept across regenerations -->', '<!-- end write-up -->'


def preserved_protocol(text):
    match = re.search(r'<!-- protocol:start -->(.*?)<!-- protocol:end -->', text, re.S)
    return match[1].strip() if match else ''


def summary_table(rows, seasons, baseline=None, note=''):
    """Rows contain saved per-season means, never values inferred from figures."""
    seasons = seasons or ['2023-2024', '2024-2025', '2025-2026']
    def fmt(value):
        return f'{value:.4f}' if value is not None else 'No known result'
    lines = ['Top three configurations by the reported combined score (all configurations if fewer than three). '
             'Season columns are seed means. Lower is better; 1 is the matched Hub ensemble.', '',
             '| Model | ' + ' | '.join(seasons) + ' | Combined |',
             '|---|' + '---:|' * (len(seasons) + 1)]
    for row in rows:
        lines.append('| ' + row['model'].replace('|', '/') + ' | ' +
                     ' | '.join(fmt(row.get(s)) for s in seasons) + ' | ' + fmt(row.get('combined')) + ' |')
    if not rows:
        lines.append('| No known model-level scores | ' + ' | '.join(['No known result'] * (len(seasons)+1)) + ' |')
    lines.append('| Hub ensemble | ' + ' | '.join(['1.0000'] * (len(seasons)+1)) + ' |')
    baseline = baseline or {}
    # A baseline combined value is only supplied for the same set of seasons.
    lines.append('| B0 reference | ' + ' | '.join(fmt(baseline.get(s)) for s in seasons) +
                 ' | ' + fmt(baseline.get('combined')) + ' |')
    lines += ['', note or 'No known comparable B0 season scores. Missing entries are not inferred from plots.', '']
    return '\n'.join(lines)


def ranking_summary(ranking, baseline_path=None):
    import numpy as np
    import pandas as pd
    ranking = Path(ranking)
    ranked = pd.read_csv(ranking / 'configuration_ranking.csv', keep_default_na=False).sort_values('combined_mean').head(3)
    scores = pd.read_csv(ranking / 'season_composite_scores.csv', keep_default_na=False)
    scores = scores[scores.geography.eq('all')]
    seasons = sorted(scores.season.unique())
    rows = []
    for i, row in enumerate(ranked.itertuples(), 1):
        part = scores[scores.config_id.eq(row.config_id)].groupby('season').combined.mean()
        rows.append(dict(model=f'C{i}', combined=float(row.combined_mean), **part.to_dict()))
    baseline = {}
    note = 'C1–C3 refer to the full ranking below. No known comparable B0 season scores.'
    if baseline_path and Path(baseline_path).exists():
        b = pd.read_csv(baseline_path)
        b = b[b.config_id.eq('stage09') & b.geography.eq('all') & b.season.isin(seasons)]
        baseline = b.groupby('season').combined.mean().to_dict()
        if set(baseline) == set(seasons):
            baseline['combined'] = float(np.mean(list(baseline.values())))
        note = ('C1–C3 refer to the full ranking below. B0 is the reproduced stage09 reference, '
                'averaged over these same seasons and its three seeds. B0 uses finalized target histories '
                'and a different training recipe and forecast sample count; this is a reference comparison, '
                'not a matched intervention or a claim of significance.')
    return summary_table(rows, seasons, baseline, note)


def figure(path, caption, width=None, note=''):
    """One report figure; `note` (the standard ★ input footnote) is printed under the graph."""
    caption = caption.replace('\n', ' ').replace('`', '')
    caption = caption[:160] if len(caption) > 160 else caption
    style = f' style="width: {width}px"' if width else ''
    note = f'<p class="figure-note">{html.escape(note)}</p>\n\n' if note else ''
    return (f'<figure class="report-figure" markdown="1">\n\n'
            f'<figcaption>{html.escape(caption)}</figcaption>\n\n'
            f'<div class="report-plot" markdown="1">\n\n'
            f'![{caption}]({path}){{{style.strip()}}}\n\n</div>\n\n'
            f'{note}[Open original figure]({path})\n\n</figure>')


def organize_report(page, summary=None, protocol=None):
    """Reorder a report, extract embedded PNGs byte-for-byte, and retain supplementary figures.

    No model loading, scoring or plotting. Stored section markers preserve an
    inline protocol when the canonical report writer regenerates its body.
    """
    import struct
    page = Path(page)
    text = page.read_text()
    meta_path = page.parent / 'report.json'
    meta = json.loads(meta_path.read_text())
    protocol = protocol if protocol is not None else preserved_protocol(text)
    text = re.sub(r'<!-- protocol:start -->.*?<!-- protocol:end -->', '', text, flags=re.S)
    text = text.replace(START, '').replace(END, '')
    text = re.sub(r'<figure class="report-figure".*?</figure>', '', text, flags=re.S)
    buckets = {s: [] for s in SECTIONS}
    title, _, body = text.partition('\n')
    blocks = re.split(r'^## (.+)\n', body, flags=re.M)
    if not protocol:
        preamble = blocks[0].strip()
        buckets['Findings' if len(preamble) > 800 else 'Protocol'].append(preamble)
    gallery = []
    seen = set()
    images = re.compile(r'!\[([^\]]*)\]\((data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+|[^\s)]+)\)')

    def place_image(match, section, heading):
        alt, url = match.groups()
        if url.startswith('data:'):
            raw = base64.b64decode(url.split(',', 1)[1])
            digest = hashlib.sha256(raw).hexdigest()[:16]
            dest = page.parent / 'figures' / f'{digest}.png'
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(raw)
            url = dest.relative_to(page.parent).as_posix()
        else:
            dest = page.parent / url
        if not dest.exists():
            return match[0]
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest in seen:
            return ''
        seen.add(digest)
        try:
            header = dest.read_bytes()[:24]
            if header[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Not a PNG')
            width, height = struct.unpack('>II', header[16:24])
            # Panoramic timelines keep readable detail via horizontal scrolling;
            # tall ranking charts scroll vertically instead of stretching the page.
            display_width = min(width, 4000 if width > 4 * height else 1200)
        except Exception:
            display_width = 1200
        target = section if section == 'Season splits' else 'Forecast fans' if re.search('fan|forecast', alt+' '+heading, re.I) else 'Score diagnostics'
        item = dict(path=url, caption=alt or heading, section=target, width=display_width)
        gallery.append(item)
        return ''

    for heading, content in zip(blocks[1::2], blocks[2::2]):
        if heading == 'Best models by season':
            continue
        if heading in SECTIONS:
            section = heading
        elif re.search('cross.validation|season split', heading, re.I):
            section = 'Season splits'
        elif re.search('protocol|experiment and score|snapshot and experiment|training budget|what the formulations', heading, re.I):
            section = 'Protocol'
        elif re.search('appendix|scenario', heading, re.I):
            section = 'Appendix'
        elif re.search('^standard evaluation', heading, re.I):
            section = 'Standard evaluation'
        elif re.search('^ranking', heading, re.I):
            section = 'Full ranking'
        elif re.search('matched comparisons|one.factor|paired contrasts|masking around', heading, re.I):
            section = 'Matched comparisons'
        elif re.search('fan plots|forecast fans', heading, re.I):
            section = 'Forecast fans'
        elif re.search('configurations.*ranked|WIS ratio by|coverage and forecast', heading, re.I):
            section = 'Score diagnostics'
        else:
            section = 'Findings'
        content = images.sub(lambda m: place_image(m, section, heading), content).strip()
        if content and content != '_Not written yet._' and not content.startswith('No known '):
            if heading not in SECTIONS:
                content = f'### {heading}\n\n' + re.sub(r'^(#{3,}) ', r'#\1 ', content, flags=re.M)
            buckets[section].append(content)
    # Supplemental figures survive canonical regeneration, without keeping an old analysis script.
    for item in meta.get('figures', []):
        path = page.parent / item['path']
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                gallery.append(item)
    if protocol:
        protocol = images.sub(lambda m: place_image(m, 'Protocol', 'Protocol'), protocol)
        buckets['Protocol'].append(re.sub(r'^(#{1,2}) ', lambda m: '#' * (len(m[1])+2) + ' ', protocol, flags=re.M))
    note_path = page.parent / 'standard-note.txt'
    note = note_path.read_text().strip() if note_path.exists() else ''
    for item in gallery:
        if 'standard' in item['path'] or 'standard' in item['caption'].lower():
            item['section'] = 'Standard evaluation'
        buckets[item['section']].append(figure(item['path'], item['caption'], item['width'], note))
    meta['figures'] = gallery
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    summary_path = page.parent / 'season-summary.txt'
    if summary is not None:
        summary_path.write_text(summary)
    elif summary_path.exists():
        summary = summary_path.read_text()
    else:
        summary = summary_table([], [])
    buckets['Best models by season'] = [summary]
    lines = [title, '']
    if meta.get('archived'):
        lines += ['Archived experiment. Results and figures describe the recorded protocol; current execution instructions are in the Workflow page.', '']
    for section in SECTIONS:
        content = '\n\n'.join(x.strip() for x in buckets[section] if x.strip()) or 'No known '+ ('season-split graph or description.' if section == 'Season splits' else 'standard evaluation: runs were not scored on standard reported inputs.' if section == 'Standard evaluation' else 'result or figure.')
        if section == 'Findings':
            content = f'{START}\n\n{content}\n\n{END}'
        elif section == 'Protocol':
            content = f'<!-- protocol:start -->\n\n{content}\n\n<!-- protocol:end -->'
        lines += [f'## {section}', '', content, '']
    page.write_text('\n'.join(lines))
