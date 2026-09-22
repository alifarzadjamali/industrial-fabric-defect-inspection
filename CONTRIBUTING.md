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

## Documentation-only updates

For maintenance changes that do not alter the experiment, keep the scope
limited to wording, navigation, or reproducibility notes. Do not revise the
reported metrics or locked evaluation protocol as part of a documentation-only
change. If a result or command is genuinely updated later, describe the reason
and the validation performed in the pull request.
