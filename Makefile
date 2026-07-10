.PHONY: check fix lint format typecheck

# Read-only gate: lint, verify formatting, and type-check. Mutates nothing.
# Suitable for CI / pre-push. Fails on the first tool that reports a problem.
check: lint format typecheck

# Apply every safe autofix: lint fixes then reformat in place.
fix:
	uv run ruff check --fix src
	uv run ruff format src

lint:
	uv run ruff check src

format:
	uv run ruff format --check src

typecheck:
	uv run basedpyright
