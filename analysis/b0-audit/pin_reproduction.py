"""Pin untouched B0 source and its saved parameter configurations for reproduction."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
root=Path('data/experiments')/(sys.argv[1] if len(sys.argv)>1 else 'b0-exact-reproduction')
old=Path('data/experiments/B0.1')
source=old/'code/src'
dest=root/'legacy-code/src'
assert not list(root.glob('*/s*/attempt-*')), 'Do not repin an already launched reproduction'
shutil.copytree(source,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
selected=json.loads(Path('data/audits/b0/selected.json').read_text())
configs={};checks=[]
for row in selected:
    if row['seed']!=42:continue
    m=json.loads((Path(row['path'])/'manifest.json').read_text())
    assert hashlib.sha256(Path(m['config']['dataset']).read_bytes()).hexdigest()==m['dataset_sha256']
    for name,expected in m['code_sha256'].items():
        path=dest/Path(name).relative_to(Path('/proj/jlessler/projects/tapestry-all/tapestry')/source)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==expected,(path,actual,expected)
        checks.append(dict(file=str(path),sha256=actual))
    configs[m['config']['fit_partition']]=m['config']
populations = None
for row in selected:
    manifest = json.loads((Path(row['path'])/'manifest.json').read_text())
    for fold in manifest['folds']:
        for component in fold['components']:
            values = component['experiment']['populations']
            if populations is None:
                populations = values
            assert values == populations, 'Historical population values differ'
population_file = Path('data/audits/b0/populations-from-checkpoint.csv')
with population_file.open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['location', 'abbreviation', 'population'])
    writer.writeheader()
    writer.writerows(dict(location=k, abbreviation=k, population=v) for k, v in populations.items())
(root/'legacy-configs.json').write_text(json.dumps(configs,indent=2)+'\n')
(root/'legacy-source-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
p=root/'code/src/tapestry/experiment/planner.py';s=p.read_text();anchor="    if args.command == 'fit':\n"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"        from tapestry.legacy_reproduction import fit as legacy_fit\n        legacy_fit(args)\n        return\n")
p.write_text(s)
shutil.copy2('analysis/b0-audit/legacy_fit.py',root/'code/src/tapestry/legacy_reproduction.py')
print('Pinned historical source; every recorded scientific source hash and dataset hash matches.')
