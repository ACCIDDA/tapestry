"""Dated experiment metadata and the model-runs index, without model dependencies."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def report_metadata(directory, name):
    path = Path(directory) / 'report.json'
    if path.exists():
        return json.loads(path.read_text())
    metadata = dict(date=datetime.now(ZoneInfo('America/New_York')).date().isoformat(),
                    title=name, stage='forecast')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def experiment_reports(root):
    reports = [(p.parent.name, json.loads(p.read_text())) for p in Path(root).glob('*/report.json')]
    return sorted(reports, key=lambda entry: (entry[1]['date'], entry[0]), reverse=True)


def write_index(root='docs/experiments'):
    root = Path(root)
    lines = ['# Model runs', '',
             'Reports are ordered newest first. Dates identify the report, not necessarily the final training job. '
             'The Kinsa report date is its first recorded publication in this repository. '
             'Use the [workflow](../workflow.md) to plan, resume and analyze experiments.', '']
    for stage, heading in [('nowcast', 'Nowcasting'), ('forecast', 'Forecasting')]:
        lines += [f'## {heading}', '']
        reports = [(name, meta) for name, meta in experiment_reports(root) if meta['stage'] == stage]
        if not reports:
            lines += ['No experiments reported yet.', '']
        else:
            lines += [f'- [{meta["date"]} · {meta["title"]}]({name}/index.md)' for name, meta in reports]
            lines.append('')
    if (root / 'archive').exists():
        archived = experiment_reports(root / 'archive')
        archive_lines = ['# Archived experiments', '',
            'Historical experiment results and original figures. Protocols describe those runs; use the '
            '[current workflow](../../workflow.md) for execution. Dates identify first recorded publication in Git.', '']
        archive_lines += [f'- [{meta["date"]} · {meta["title"]}]({name}/index.md)' for name, meta in archived]
        (root / 'archive' / 'index.md').write_text('\n'.join(archive_lines) + '\n')
        lines += ['## Archive', '', '[Archived experiments](archive/index.md). These are historical results, not current workflow instructions.', '']
    (root / 'index.md').write_text('\n'.join(lines))
