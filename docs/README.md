# UAEF Documentation Site

MkDocs + Material site. Built locally for review; no hosting configured yet — see
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
├── dev/                   <- internal working notes; NOT published
├── _md_to_pdf.py          <- one-off doc conversion tooling
└── _md_to_docx.py
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
- **Internal notes stay out.** `dev/` sits outside `docs/docs/`, so backlogs and design
  scratch are readable in the repo but never published.
- **Non-site files are named, not linked.** Anything outside `docs/docs/` — source under
  `src/`, root `README.md`, `SECURITY.md`, `LICENSE`, `uaef-service/` — is written as a
  plain repo-relative path in inline code, e.g. `` `src/uaef/metrics/tool_calling.py` ``.
  Relative links cannot escape `docs_dir`, and an absolute URL would hardcode a host that
  is not decided yet. Turn these into links once one is.

## Publishing

Not configured. `make build` produces a self-contained `site/` directory that any static
host can serve; no hosting location is committed here.

Whoever wires up hosting needs to:

1. Set `site_url` and `repo_url` in `mkdocs.yml`. Both are intentionally unset — the theme
   omits its repository link without `repo_url`, and leaving `site_url` unset keeps
   `mkdocs serve` on `/` instead of a base path.
2. Turn references to repo files into links. Paths to source, `SECURITY.md`, `LICENSE`, and
   `uaef-service/README.md` are written as plain inline code rather than links, because
   they live outside `docs_dir` and cannot be linked relatively. Linkifying them is a
   scripted pass once a canonical repo URL exists.
3. Install the toolchain with `uv sync --group docs --frozen`. Because of `--frozen`, run
   `uv lock` from the repo root after changing that dependency group or the build fails.
4. Run an external link check once after linkifying, e.g. `lychee 'docs/**/*.md'`.
   `make build` cannot help here — it only validates links between pages, never external
   URLs.
