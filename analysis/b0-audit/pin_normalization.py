"""Apply one explicit normalization intervention to a newly planned code snapshot.

Experimental finalized/finalized_available forecasters only; production source is untouched.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

experiment=Path('data/experiments')/sys.argv[1]
source=Path('data/audits/b0/old-src/tapestry/models/experiments.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='input_scales')
function='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
function=function.replace('from .b0 import transform_counts, transform_proportions',
                          'from tapestry.model.network import transform_counts, transform_proportions')
# For finalized-available episodes the same date can be absent at one issuance and
# present at another. Union visible finalized cells rather than overwrite with missing.
function=function.replace('by_date[day] = row', '''if day not in by_date:
                by_date[day] = row.copy()
            else:
                valid_row = row[:, 1].astype(bool)
                by_date[day][:, 0][valid_row] = row[:, 0][valid_row]
                by_date[day][:, 1][valid_row] = 1''')
p=experiment/'code/src/tapestry/experiment/training.py'
s=p.read_text();anchor='    if \'history_samples\' in batch[0]:\n'
assert s.count(anchor)==1 and 'def input_scales(' not in s
insert='''    if scenario.task != 'forecast' or scenario.input_mode not in ('finalized', 'finalized_available'):
        raise ValueError('B0 normalization audit only supports finalized forecast inputs')
    normalization_episodes = [dict(e, X=np.stack((e['values'], e['available']), axis=2)) for e in batch]
    options.update(input_scales(normalization_episodes, scenario.count_transform, scenario.ed_transform, pop))
'''
s=s.replace(anchor,insert+anchor)+ '\n\n'+function
p.write_text(s)
(experiment/'normalization-intervention.json').write_text(json.dumps(dict(
    intervention='B0 input normalization; union of visible finalized context cells by date',
    training_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    unchanged='Loss, architecture, input availability, masks, data, seed, stopping, sample count'),indent=2)+'\n')
print(experiment/'normalization-intervention.json')
