"""Pin a single intervention per consecutive B1-to-B0 arm, never production code."""
import argparse,hashlib,json,shutil
from pathlib import Path


def replace(source,old,new):
 assert source.count(old)==1,(old,source.count(old))
 return source.replace(old,new)


def main(stage):
 root=Path('data/experiments')/f'b1-b0-chain-{stage}'
 assert (root/'experiment.json').exists()
 assert not list(root.glob('*/s*/attempt-*')) and not (root/'chain-intervention.json').exists()
 reference=Path('data/experiments')/('b0-fitting-split-draws' if stage in ('07','08','09') else 'b0-current-training-normalized')/'code'
 shutil.rmtree(root/'code');shutil.copytree(reference,root/'code')
 file=root/'code/src/tapestry/experiment/training.py';s=file.read_text()
 if stage=='00':
  s=replace(s,"    if scenario.task != 'forecast' or scenario.input_mode not in ('finalized', 'finalized_available'):\n        raise ValueError('B0 normalization audit only supports finalized forecast inputs')\n    normalization_episodes = [dict(e, X=np.stack((e['values'], e['available']), axis=2)) for e in batch]\n    options.update(input_scales(normalization_episodes, scenario.count_transform, scenario.ed_transform, pop))\n",'')
 if stage in ('07','08','09'):
  s=replace(s,'    weights_by_channel = LOSS_WEIGHTS[scenario.loss_weights]\n','    weights_by_channel = np.asarray(LOSS_WEIGHTS[scenario.loss_weights]).copy()\n    weights_by_channel[[c for c in range(len(CHANNELS)) if c not in channels]] = 0\n')
 if stage in ('08','09'):
  # Restore original six-channel loss reduction layout, zeroing unsupervised labels.
  # This preserves the objective but can change GPU floating-point accumulation.
  s=replace(s,'    weights = torch.as_tensor(loss_cell_weights(train, weights_by_channel), device=device)[:, :, channels]\n','    weights = torch.as_tensor(loss_cell_weights(train, weights_by_channel), device=device)\n    y_mask = y_mask.clone()\n    y_mask[:, :, [c for c in range(len(CHANNELS)) if c not in channels]] = False\n')
  s=replace(s,'        vweights = torch.as_tensor(loss_cell_weights(validation, weights_by_channel), device=device)[:, :, channels]\n','        vweights = torch.as_tensor(loss_cell_weights(validation, weights_by_channel), device=device)\n        vy_mask = vy_mask.clone()\n        vy_mask[:, :, [c for c in range(len(CHANNELS)) if c not in channels]] = False\n')
  s=replace(s,'            score = fair_crps_cells(samples[:, :, :, channels], y[ids][:, :, channels], y_mask[ids][:, :, channels])\n','            score = fair_crps_cells(samples, y[ids], y_mask[ids])\n')
  s=replace(s,'                    score = fair_crps_cells(samples[:, :, :, channels], vy[ids][:, :, channels], vy_mask[ids][:, :, channels])\n','                    score = fair_crps_cells(samples, vy[ids], vy_mask[ids])\n')
  s=s.replace(' / model.scale[channels]', ' / model.scale')
 if stage=='09':
  s=replace(s,"    y_mask[:, :, [c for c in range(len(CHANNELS)) if c not in channels]] = False\n","    y_mask[:, :, [c for c in range(len(CHANNELS)) if c not in channels]] = False\n    packed_y = torch.stack((torch.where(y_mask, y, 0), y_mask.to(y.dtype)), dim=3)\n")
  s=replace(s,"        vy_mask[:, :, [c for c in range(len(CHANNELS)) if c not in channels]] = False\n","        vy_mask[:, :, [c for c in range(len(CHANNELS)) if c not in channels]] = False\n        packed_vy = torch.stack((torch.where(vy_mask, vy, 0), vy_mask.to(vy.dtype)), dim=3)\n")
  s=replace(s,'            score = fair_crps_cells(samples, y[ids], y_mask[ids])\n','            score = fair_crps_cells(samples, packed_y[ids, :, :, 0, :], packed_y[ids, :, :, 1, :])\n')
  s=replace(s,'                    score = fair_crps_cells(samples, vy[ids], vy_mask[ids])\n','                    score = fair_crps_cells(samples, packed_vy[ids, :, :, 0, :], packed_vy[ids, :, :, 1, :])\n')
 file.write_text(s)
 hashes=lambda tree:{str(p.relative_to(tree)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tree.rglob('*.py'))}
 old,new=hashes(reference),hashes(root/'code');changed=[k for k in new if new[k]!=old.get(k)]
 assert changed==(['src/tapestry/experiment/training.py'] if stage in ('00','07','08','09') else [])
 (root/'chain-intervention.json').write_text(json.dumps(dict(stage=stage,reference=str(reference),changed_files=changed,code_hashes=new),indent=2)+'\n')
 print(root/'chain-intervention.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['00','01','02','03','07','08','09']);main(p.parse_args().stage)
