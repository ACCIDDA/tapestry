"""Independent supervision bundles retain all six observed input channels."""
import json

import torch
from torch import nn

GROUPS = {'all': [list(range(6))], 'pathogen': [[0, 3], [1, 4], [2, 5]],
          'target': [[i] for i in range(6)]}


def supervision(episodes, channels):
    selected = []
    for episode in episodes:
        y = episode['Y'].copy()
        y[:, [c for c in range(6) if c not in channels]] = 0
        if y[:, :, 1].any():
            selected.append({**episode, 'Y': y})
    if not selected:
        raise ValueError(f'No supervision for channels {channels}')
    return selected


class IndependentBundle(nn.Module):
    def __init__(self, models, groups):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.groups = groups
        self.config = models[0].config

    def forward(self, x, calendar, members=8, **kwargs):
        # Each component consumes fresh draws; matching member indices imply no
        # learned cross-component dependence. Within-component paths stay intact.
        outputs = [model(x, calendar, members, **kwargs)[:, :, :, group]
                   for model, group in zip(self.models, self.groups)]
        order = [c for group in self.groups for c in group]
        return torch.cat(outputs, 3)[:, :, :, [order.index(c) for c in range(6)]]


def checkpoint(model, metadata):
    # JSON converts numpy string scalars from fold dates to weights-only-safe types.
    metadata = json.loads(json.dumps(metadata))
    if isinstance(model, IndependentBundle):
        return dict(groups=model.groups, components=[checkpoint(m, {}) for m in model.models], metadata=metadata)
    return dict(config=model.config, state_dict={k: v.cpu() for k, v in model.state_dict().items()}, metadata=metadata)


def load_model(saved):
    from .b0 import B0
    if 'components' in saved:
        return IndependentBundle([load_model(c) for c in saved['components']], saved['groups'])
    model = B0(**saved['config'])
    model.load_state_dict(saved['state_dict'])
    return model
