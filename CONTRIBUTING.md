# Contributing

Thanks for helping improve the fabric inspection project.

## Development setup

Create the environment described in the [README](README.md), then install the
development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run the focused checks before opening a change:

```powershell
python -m pytest
ruff check src tests scripts
python -m pip check
```

## Pull requests

- Keep changes focused and explain the reason for the change.
- Do not commit downloaded datasets, model checkpoints, or generated local data.
- Update the relevant documentation when a command, configuration, or result changes.
- Include test coverage for behavior changes where practical.
