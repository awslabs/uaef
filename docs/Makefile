# Makefile for the UAEF documentation site.
#
# The site is MkDocs + Material, with mkdocstrings for API reference and
# awesome-nav for navigation generated from the directory tree.
#
# All targets run from this directory (docs/). Dependencies come from the
# `docs` dependency group in ../pyproject.toml.

.PHONY: install docs build clean

# install: Install the documentation toolchain from the lockfile.
#
# `--frozen` fails rather than silently re-resolving, which keeps local builds
# identical to CI. If it errors because the lockfile is stale, run `uv lock`
# from the repo root.
install:
	uv sync --group docs --frozen

# docs: Serve locally on http://127.0.0.1:8000 with live reload.
docs:
	uv run mkdocs serve --livereload

# build: Build the static site into site/, failing on broken links.
#
# Run this before opening a PR. See check_build.sh for why this is not a plain
# `mkdocs build --strict`.
build:
	./check_build.sh

# No deploy target: no hosting location is configured yet. `make build` produces a
# self-contained site/ directory that any static host can serve.

# clean: Remove the built site.
clean:
	rm -rf site/
