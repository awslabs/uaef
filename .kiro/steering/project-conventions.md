# Project Conventions

## Package Management

This project uses **uv** as the package manager. Do NOT use `pip` directly.

```bash
# Install a package
uv pip install <package>

# Install from requirements
uv pip install -r requirements.txt

# The venv is at .venv/ but has no pip binary — always use uv
```

## Python Environment

- Python 3.12 (managed via mise)
- Virtual environment: `.venv/`
- Source code: `src/uaef/`
- Notebooks: `notebooks/`
- Reports output: `output/reports/` (gitignored)

## Testing Notebooks

When testing notebook code from the command line, use:
```bash
.venv/bin/python -c "..."
```
Or for scripts that need the src path:
```bash
python3 -c "import sys; sys.path.insert(0, 'src'); ..."
```
