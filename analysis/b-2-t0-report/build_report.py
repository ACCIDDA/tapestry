"""Complete standard report, selecting the three best configurations for fans/heatmaps."""
from pathlib import Path
import pandas as pd
from tapestry.experiment.planner import rank
from tapestry.evaluation.plots import plot_experiment, write_report
folder = Path('data/experiments/b-2-t0')
ranking = rank(folder, make_plots=False)
configs = pd.read_csv(ranking / 'configuration_ranking.csv').head(3).config_id.tolist()
for path in plot_experiment(folder, ranking, configs=configs):
    print(path, flush=True)
print(write_report(folder, ranking), flush=True)
Path('analysis/b-2-t0-report/ranking-path.txt').write_text(str(ranking) + '\n')
