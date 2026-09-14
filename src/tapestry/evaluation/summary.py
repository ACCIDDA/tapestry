"""Select comparable competitors from saved R scores; never rescore or refit."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .hubs import KEY

METRICS = ['wis', 'ae_median', 'interval_coverage_50', 'interval_coverage_95',
           'overprediction', 'underprediction', 'dispersion', 'bias']


def select_best(scores, model_name, ensemble):
    board = scores.groupby('model')[METRICS].mean()
    board['n'] = scores.groupby('model').size()
    total = int(board.loc[ensemble, 'n'])
    board['coverage'] = board.n / total
    submitted = ~board.index.isin([model_name, ensemble]) & ~board.index.str.lower().str.contains('baseline')
    eligible = list(board.index[submitted & (board.n == total)])
    fallback = not eligible
    if fallback:
        eligible = list(board.index[submitted & (board.coverage >= .90)])
    units = scores[scores.model == ensemble][KEY].copy()
    for model in eligible:
        units = units.merge(scores[scores.model == model][KEY], on=KEY, validate='one_to_one')
    restricted = scores.merge(units, on=KEY, validate='many_to_one')
    ranked = restricted[restricted.model.isin(eligible)].groupby('model').wis.mean().sort_values()
    winner = str(ranked.index[0]) if len(ranked) else None
    board['eligible_best'] = board.index.isin(eligible)
    board['ranking_wis_common'] = ranked
    models = [m for m in [model_name, winner, ensemble] if m]
    display = restricted[restricted.model.isin(models)]
    summary = display.groupby('model')[METRICS].mean()
    summary['n'] = display.groupby('model').size()
    return board, winner, units, summary.reset_index().to_dict('records'), fallback


def refresh(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    for info in manifest['cases']:
        if info['status'] != 'scored':
            continue
        folder = output / info['directory']
        scores = pd.read_csv(folder / 'scores.csv', dtype={'location': str})
        board, best, units, summary, fallback = select_best(scores, info['model_name'], info['ensemble'])
        board.sort_values('wis').to_csv(folder / 'leaderboard.csv')
        units.to_parquet(folder / 'comparison_units.parquet', index=False)
        info.update(best=best, summary=summary, n_comparison_units=len(units),
                    best_coverage_fallback=fallback,
                    eligible_best_models=list(board.index[board.eligible_best]),
                    best_rule=('No complete competitor: >=90% coverage; rank on identical common ensemble units' if fallback else
                               '100% coverage competitors; rank on identical full ensemble units'))
        (folder / 'comparison.json').write_text(json.dumps(info, indent=2) + '\n')
    manifest['best_rule'] = 'Prefer complete submitted competitors; if none, >=90% coverage and identical shared tasks. Full ensemble-supported R scores always retained.'
    manifest['summary_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = [{**{k: case[k] for k in ['hub', 'season', 'target', 'n_units', 'n_comparison_units', 'best_coverage_fallback']}, **row}
            for case in manifest['cases'] if case['status'] == 'scored' for row in case['summary']]
    pd.DataFrame(rows).to_csv(output / 'comparison_summary.csv', index=False)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    refresh(parser.parse_args().output)
