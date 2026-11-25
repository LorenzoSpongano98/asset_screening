check:
  uv lock --check 
  uv sync --locked
  uv run ruff check .
  uv run ruff format --check . 
  uv run pyright
  uv run pytest -m "not network" --cov=asset_screening --cov-report=term-missing
fmt:
  uv run ruff check .
  uv run ruff format --check . 

