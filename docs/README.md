# UAEF Documentation Site

MkDocs + Material site, published to <https://awslabs.github.io/uaef/> — see
[Publishing](#publishing).

## Layout

```
docs/                      <- you are here; mkdocs.yml lives at this level
├── mkdocs.yml
├── Makefile
├── docs/                  <- page sources (the MkDocs docs_dir)
│   ├── .nav.yml           <- top-level nav; each subdirectory is a tab
│   ├── index.md
│   ├── Getting-Started/
│   ├── Guides/
│   ├── Advanced/
│   ├── API-Reference/     <- `::: uaef.<module>` directives, no hand-written API prose
│   └── Contributing/
├── check_build.sh         <- strict-ish build check; run by `make build` and by CI
└── .gitignore             <- ignores the built site/
```

The `docs/docs/` nesting is deliberate: CI runs `cd docs && mkdocs gh-deploy`, which
requires `mkdocs.yml` to sit one level above the page sources.

## Working on it

```bash
make install     # uv sync --group docs --frozen
make docs        # serve at http://127.0.0.1:8000, live reload
make build       # strict build; fails on broken links and bad mkdocstrings refs
make clean
```

Run `make build` before opening a PR. It catches the failure mode this site is most prone
to: a relative link that broke when a page moved.

`make build` runs `check_build.sh` rather than `mkdocs build --strict`, because strict mode
also escalates griffe's "No type or annotation for parameter …" warnings — there are
currently 32 of them, all from missing annotations in `src/uaef` (mostly undocumented
`**kwargs`). Those are real gaps worth closing in the library, but they are not docs
defects and should not block docs changes. Once the annotations land, swap `check_build.sh`
for `mkdocs build --strict`.

## Conventions

- **Navigation** comes from the directory tree via `mkdocs-awesome-nav`. Order pages by
  listing them in the directory's `.nav.yml`; set the tab label with `title:` in the same
  file. Adding a page means adding one line there.
- **API reference is generated.** Pages under `API-Reference/` contain only a short intro
  and `::: uaef.<module>` directives resolved against `../src`. Document new functions with
  docstrings, not by editing these pages. A new *module* needs a directive added.
- **Everything under `docs/docs/` is published.** There is no unpublished scratch area;
  design notes and backlogs belong in issues, not in this directory.
- **Non-site files link to GitHub.** Anything outside `docs/docs/` — source under `src/`,
  `SECURITY.md`, `LICENSE`, `uaef-service/` — cannot be linked relatively, because relative
  links cannot escape `docs_dir`. Write it as an absolute link on `main` with the path as
  the link text, e.g.
  ``[`src/uaef/metrics/tool_calling.py`](https://github.com/awslabs/uaef/blob/main/src/uaef/metrics/tool_calling.py)``.
  Use `/blob/main/` for files and `/tree/main/` for directories. `make build` does not
  verify these, so check the path exists before committing.

## Publishing

GitHub Pages, via `.github/workflows/docs.yml`:

- **Pull requests** touching `docs/`, `src/`, `pyproject.toml`, or `uv.lock` run
  `check_build.sh` — the same check as `make build`. Nothing is published.
- **Pushes to `main`** run that build, then `mkdocs gh-deploy --force`, which commits the
  built site to the `gh-pages` branch. Pages serves it at
  <https://awslabs.github.io/uaef/>.

Never deploy from a laptop. `gh-deploy` force-pushes `gh-pages`, so a local run publishes
whatever happens to be in the working tree.

The site is public while the repository is private. That is the GitHub Pages default and it
is intentional — the docs went up ahead of the repository being opened. Two consequences
worth knowing:

- Anything reachable from `docs/docs/` is world-readable once deployed. `dev/` sits outside
  it and is not published.
- The generated API reference embeds library source, because mkdocstrings runs with
  `show_source: true`. Publishing the site publishes that source.

### One-time repository settings

1. **Settings → Pages → Source: Deploy from a branch**, branch `gh-pages`, folder `/`. The
   branch is created by the first `deploy` run, so push to `main` first, then set this.
2. **Settings → Actions → General → Workflow permissions**: the `deploy` job requests
   `contents: write` itself, so read-only defaults are fine. "Allow GitHub Actions to
   create and approve pull requests" is not needed and should stay off.

After the first deploy, load <https://awslabs.github.io/uaef/> and check a mermaid diagram,
the search box, and a couple of the absolute `blob/main/` links. The first two are what a
`site_url` base-path change breaks; the third CI never verifies. Those `blob/main/` links
404 for anonymous visitors until the repository itself is public.

### Still open

- No external link check in CI. `make build` only validates links between pages, so the
  absolute GitHub links described under [Conventions](#conventions) are unverified. Run
  something like `lychee 'docs/docs/**/*.md'` periodically, or add it as a scheduled job.
- The docs toolchain installs with `--frozen`, so run `uv lock` from the repo root after
  changing the `docs` dependency group or CI fails.
