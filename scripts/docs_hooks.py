"""Keep the MkDocs experiment navigation ordered by the reports' explicit dates."""
import json
from pathlib import Path


def on_config(config):
    root = Path(config['docs_dir']) / 'experiments'
    reports = [(p.parent.name, json.loads(p.read_text())) for p in root.glob('*/report.json')]
    reports.sort(key=lambda entry: (entry[1]['date'], entry[0]), reverse=True)
    for section in config['nav']:
        if 'Experiments' in section:
            children = [{'Model runs': 'experiments/index.md'}]
            for stage, label in [('nowcast', 'Nowcasting'), ('forecast', 'Forecasting')]:
                pages = []
                for name, meta in reports:
                    if meta['stage'] != stage:
                        continue
                    title = f'{meta["date"]} · {meta["title"]}'
                    pages.append({title: f'experiments/{name}/index.md'})
                if pages:
                    children.append({label: pages})
            archived = [(p.parent.name, json.loads(p.read_text())) for p in (root / 'archive').glob('*/report.json')]
            archived.sort(key=lambda entry: (entry[1]['date'], entry[0]), reverse=True)
            if archived:
                children.append({'Archive': [{'Overview': 'experiments/archive/index.md'},
                    *[{f'{meta["date"]} · {meta["title"]}': f'experiments/archive/{name}/index.md'}
                      for name, meta in archived]]})
            section['Experiments'] = children
    return config
