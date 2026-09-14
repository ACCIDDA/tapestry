# Python API

The public acquisition types are re-exported by `influpaintx.data`. Fetcher and
explorer APIs are documented separately because most consumers should interact
with them through the command line.

## Data package

::: influpaintx.data
    options:
      members_order: source
      show_root_full_path: false

## Repository

::: influpaintx.data.repository.RawDataRepository
    options:
      members: true
      show_root_full_path: false

## Data models

::: influpaintx.data.models
    options:
      members_order: source
      show_root_full_path: false

## Geography

::: influpaintx.data.geography
    options:
      members_order: source
      show_root_full_path: false

## Shared post-intake selection

::: influpaintx.data.selection.SelectedData
    options:
      members: true
      show_root_full_path: false

## Explorer index

::: influpaintx.explorer.ExplorerIndex
    options:
      members: true
      show_root_full_path: false
