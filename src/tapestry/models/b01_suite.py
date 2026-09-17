"""Expand the checked-in B0.1 specification; preserve every contrast membership."""
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

from .scenarios import TrainingScenario

SPEC = Path(__file__).resolve().parents[3] / 'docs/design/b0.1.json'


def scenario(recipe, spec):
    options = {key: value for key, value in recipe.items()
               if key not in ('output', 'exchange', 'us_heads', 'shared_factor')}
    options.update(spatial={'shared_spatial': 'attention'}.get(recipe['exchange'], recipe['exchange']),
                   heads='state_us' if recipe['us_heads'] == 'separate' else 'shared',
                   noise='local' if recipe['noise'] == 'global_local' else 'global',
                   us_error='shared_factor' if recipe['shared_factor'] else 'none',
                   members=spec['training_members'], validation_members=spec['validation_members'],
                   batch_size=spec['batch_size'], loss_weights='objective')
    return TrainingScenario(**options)


def expand():
    spec = json.loads(SPEC.read_text())
    defaults, blocks = spec['defaults'], spec['blocks']
    refs = {name: {**defaults, **recipe} for name, recipe in blocks['A_reference_crosses']['references'].items()}
    unique, entries, added = {}, Counter(), Counter()

    def add(block, name, recipe, **membership):
        candidate = scenario(recipe, spec)
        entries[block] += 1
        if candidate not in unique:
            unique[candidate] = dict(name=f'B0.1/{name}', recipe=recipe, memberships=[])
            added[block] += 1
        unique[candidate]['memberships'].append(dict(block=block, **membership))

    block = 'A_reference_crosses'
    # References first, preserving their readable aliases.
    for name, recipe in refs.items():
        add(block, name, recipe, reference=name, axis='reference')
    for name, recipe in refs.items():
        for axis, levels in blocks[block]['axes'].items():
            for value in levels:
                if value != recipe[axis]:
                    add(block, f'{name}__{axis}_{value}', {**recipe, axis: value}, reference=name, axis=axis, value=value)
    for block in ('B_attention_sharing_panels', 'C_transform_panels'):
        axes = blocks[block]['axes']
        for name in blocks[block]['reference_names']:
            for values in product(*axes.values()):
                changes = dict(zip(axes, values))
                add(block, name + '__' + '__'.join(f'{k}_{v}' for k, v in changes.items()),
                    {**refs[name], **changes}, reference=name, levels=changes)
    block = 'D_conv_controls'
    for i, recipe in enumerate(blocks[block]['configurations']):
        add(block, f'conv_control_{i}', recipe, control=i)
    block = 'E_independent_sharing_controls'
    axes = blocks[block]['axes']
    for values in product(*axes.values()):
        changes = dict(zip(axes, values))
        add(block, 'independent__' + '__'.join(f'{k}_{v}' for k, v in changes.items()), {**defaults, **changes}, levels=changes)
    if dict(entries) != spec['candidate_entries_by_block'] or dict(added) != spec['additional_unique_configurations_by_block']:
        raise ValueError(f'B0.1 grid counts changed: entries={dict(entries)}, additional={dict(added)}')
    if len(unique) != spec['unique_configurations']:
        raise ValueError('B0.1 unique configuration count mismatch')
    return spec, unique


def scenarios():
    return {record['name']: candidate for candidate, record in expand()[1].items()}


def manifest():
    spec, unique = expand()
    return dict(experiment='B0.1', seeds=spec['seeds'],
                configurations=[dict(**record, scenario=candidate.scenario_string, config=asdict(candidate))
                                for candidate, record in unique.items()])
