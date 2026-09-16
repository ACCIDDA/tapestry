"""Measure actual B0.1 training/validation shapes on an allocated GPU before packing lanes."""
import argparse
import json
from pathlib import Path
import time

import torch

from tapestry.model_data import FinalizedDataset
from tapestry.models.b0 import B0, fair_crps_cells
from tapestry.models.b01_suite import scenarios
from tapestry.models.experiments import model_options
from tapestry.models.run import tensors, validation_draws, validation_loss
from tapestry.models.season_cv import build_parser, fold_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    ds = FinalizedDataset.load('data/processed/build_b_finalized.npz')
    names = ['B0.1/joint_trend__width_128', 'B0.1/joint_trend__lookback_26',
             'B0.1/joint_multiscale__width_128', 'B0.1/joint_multiscale__lookback_26']
    recipes, results = scenarios(), []
    for name in names:
        fit_args = build_parser().parse_args(recipes[name].flags())
        fit_args.device = args.device
        training, _, scales = fold_data(ds, '2025-2026', fit_args.lookback)
        batch = training[-fit_args.batch_size:]
        model = B0(fit_args.lookback, width=fit_args.width, scale=scales,
                   **model_options(training, fit_args)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=fit_args.lr)
        x, y, cal = tensors(batch, model, args.device)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        before = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            samples = model(x, cal, fit_args.members, locations=batch[0]['locations'])
            loss = (fair_crps_cells(samples, y[:, :, :, 0], y[:, :, :, 1]) / model.scale).mean()
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss: {name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        validation_loss(model, batch, fit_args, validation_draws(model, batch, fit_args))
        if args.device == 'cuda':
            torch.cuda.synchronize()
        result = dict(name=name, scenario=recipes[name].scenario_string,
                      parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter() - before,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None)
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, samples, loss, x, y, cal
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(device=args.device, recipes=results,
        note='Two optimizer steps and one 256-member validation batch; measure throughput before increasing lanes.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
