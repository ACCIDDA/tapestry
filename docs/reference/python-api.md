# Python API

The public acquisition types are re-exported by `chromantis.data`. Fetcher and
explorer APIs are documented separately because most consumers should interact
with them through the command line.

## Data package

::: chromantis.data
    options:
      members_order: source
      show_root_full_path: false

## Repository

::: chromantis.data.repository.RawDataRepository
    options:
      members: true
      show_root_full_path: false

## Data models

::: chromantis.data.models
    options:
      members_order: source
      show_root_full_path: false

## Geography

::: chromantis.data.geography
    options:
      members_order: source
      show_root_full_path: false

## Shared post-intake selection

::: chromantis.data.selection.SelectedData
    options:
      members: true
      show_root_full_path: false

## Explorer index

::: chromantis.explorer.ExplorerIndex
    options:
      members: true
      show_root_full_path: false
