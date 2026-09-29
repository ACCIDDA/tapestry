"""Finalize shared origin calendar in unlaunched reforecast snapshots."""
from pathlib import Path
import json,hashlib,shutil
adapter=Path('analysis/b0-audit/chain_reforecast.py')
for root in sorted(Path('data/experiments').glob('b1-b0-score-*')):
 assert not list(root.glob('*/s*/attempt-*')),f'Already launched: {root}'
 shutil.copy2(adapter,root/'code/src/tapestry/chain_reforecast.py')
 protocol=json.loads((root/'reforecast-protocol.json').read_text());protocol.update(evaluation_calendar='Common full-finalized B0 origin calendar; unavailable context cells stay missing; do not skip empty contexts',adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest())
 (root/'reforecast-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('Pinned common-origin fixed-draw evaluation in 11 unlaunched snapshots')
