# Python API

The public acquisition types are re-exported by `tapestry.data`. Fetcher and
explorer APIs are documented separately because most consumers should interact
with them through the command line.

## Data package

::: tapestry.data
    options:
      members_order: source
      show_root_full_path: false

## Repository

::: tapestry.data.repository.RawDataRepository
    options:
      members: true
      show_root_full_path: false

## Data models

::: tapestry.data.models
    options:
      members_order: source
      show_root_full_path: false

## Geography

::: tapestry.data.geography
    options:
      members_order: source
      show_root_full_path: false

## Shared post-intake selection

::: tapestry.data.selection.SelectedData
    options:
      members: true
      show_root_full_path: false

## Explorer index

::: tapestry.explorer.ExplorerIndex
    options:
      members: true
      show_root_full_path: false
