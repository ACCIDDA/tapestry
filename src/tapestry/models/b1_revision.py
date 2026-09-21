"""Training-only revision augmentation, built after fold/inner-validation masking.

A donor is one complete issuance block (both ages, all channels and locations).
Only genuine reports with permitted recent labels supply residuals. Recipient
finals plus donor report-minus-final errors replace eligible provisional inputs;
we do not add another full reporting bias to an already preliminary observation.
"""
import torch
from .b0 import invert_counts


class RevisionAugmenter:
    def __init__(self, model, values, available, known_final, recent_truth, recent_valid):
        self.model = model
        self.eligible = available[:, -2:] & ~known_final[:, -2:] & recent_valid.bool()
        # Remove hidden/unlabeled values before transforming, including NaN sentinels.
        reports = torch.where(self.eligible, values[:, -2:], 0.)
        finals = torch.where(self.eligible, recent_truth, 0.)
        self.final = model.normalized(finals)
        self.errors = torch.where(self.eligible, model.normalized(reports) - self.final, 0.)
        self.donors = self.eligible.flatten(1).any(1).nonzero().flatten().cpu().numpy()
        self.support = self.eligible.sum((0, 3)).cpu().tolist()  # age, channel

    def apply(self, values, dropout, rng, rate):
        import numpy as np
        donor_ids = np.full(len(values), -1, dtype=np.int64)
        if not len(self.donors) or rate == 0:
            return values, donor_ids, 0
        chosen = rng.random(len(values)) < rate
        donor_ids[chosen] = rng.choice(self.donors, size=int(chosen.sum()))
        ids = torch.as_tensor(donor_ids.clip(min=0), device=values.device)
        eligible = (self.eligible & ~dropout[:, -2:] & self.eligible[ids]
                    & torch.as_tensor(chosen, device=values.device)[:, None, None, None])
        # Add errors in normalized transformed units. Same location/target/age
        # on both sides; one donor preserves the available cross-cell dependence.
        transformed = (self.final + self.errors[ids]) * self.model.input_scale + self.model.input_offset
        count = invert_counts(transformed[:, :, :3].clamp_min(0), self.model.population,
                              self.model.config['count_transform'])
        ed = transformed[:, :, 3:]
        transform = self.model.config['ed_transform']
        ed = ed.sigmoid() if transform == 'logit' else ed.clamp_min(0).pow(4).clamp(max=1) if transform == 'fourth_root' else ed.clamp(0, 1)
        synthetic = torch.cat((count, ed), 2)
        augmented = values.clone()
        augmented[:, -2:] = torch.where(eligible, synthetic, values[:, -2:])
        return augmented, donor_ids, int(eligible.sum())
