---
name: benchmantic
description: Turns a simulation benchmark's source code into a machine-readable description (JSON-LD -- parameters, configurations, metrics, units, inferred by an LLM and reviewed by a person), a container-ready Snakemake workflow, and optionally a benchmark catalog repository on GitHub. Supports DuMux, OpenFOAM and FEniCSx (others via an LLM-written software description). Use when a user wants a benchmark.jsonld from code, to verify or review such a description, to add another software implementation of an existing benchmark, to build its Snakefile, or to create a benchmark repository like Simulation-Benchmarks/rotating-cylinders.
license: MIT
compatibility: Python 3.10+ (pip install -r requirements.txt). An LLM -- an API key (Groq, Claude, OpenAI) or the claude-code/codex CLIs -- unless the implementation's reviewed answers are stored with it. The review needs a terminal unless --skip-review. Running the workflow needs Snakemake and Apptainer/Docker; create_benchmark_repo.py needs reuse, and gh for --github.
metadata:
  repository: https://github.com/Simulation-Benchmarks/benchmantic
  maintainer: Simulation-Benchmarks
---

# benchmantic

The README is the reference; this is the short version for running it.

## Run it

```bash
pip install -r requirements.txt
python3 workflow.py --config <run config>                 # describe -> show -> verify; exit code = verification
python3 create_benchmark_repo.py --config <run config>    # the benchmark repository (pip install reuse)
```

A run config needs `implementation` (a folder or a GitHub/GitLab link), `software`
and `container`; `examples/taylor-couette-flow/benchmantic.yaml` is a complete one
that runs without an LLM. Outputs land in `outputs/<software-name>/`
(`*_benchmark.jsonld`, `*_dataset.jsonld`, `*_mapping.json`, `Snakefile`, `review.md`).

## Unattended runs (CI, scripts)

- Pass `scenario-params` (the inputs that differ between configurations) and
  `skip-review: true`, or the run waits at a prompt.
- Pass `outputs` (e.g. `standard`) or the module's last choice is reused.
- The LLM key comes from `LLM_API_KEY` or the provider's variable (`GROQ_API_KEY`,
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); without one and without stored answers
  the run stops with a message. Never put a key into a run config.
- Treat a failing `verify_description.py` (non-zero exit of `workflow.py`) as a hard
  failure: don't publish that description.

## Stored answers

The reviewed LLM answers sit next to the implementation
(`.parameter_metadata_cache.json`, `.metric_metadata_cache.json`) and are reused:
the same answers every run, no LLM call for them. `--clear-cache` asks again. To fix
one answer, edit the file and run again without `--clear-cache`. Answers given with
`skip-review` are unreviewed; review them before committing them.

## A benchmark repository

`create_benchmark_repo.py --config <run config> [--org ACCOUNT] [--github] [--watch]`
reads the config's `repository:` block (`name`, `title`, `summary`, `documentation`,
`container` -- the Dockerfile's folder or a link to it -- `authors`, `keywords`) and
fills in `repository_template/`. `--github` creates (or updates, with one new commit)
`ACCOUNT/name` via `gh` -- the token needs the `repo` and `workflow` scopes;
`--watch` waits for its Actions. The repository includes the RoHub workflow and notebook of rotating-cylinders; the weekly run uploads to the development RoHub, manual runs too unless `rohub_production: true`. `--rohub` sets RoHub up: from `ROHUB_DEV_USERNAME`/`ROHUB_DEV_PASSWORD` (and `ROHUB_PROD_*`) in the environment it finds or creates the benchmark's research object (or uses `rohub_dev_identifier`/`rohub_prod_identifier`), stores the logins and the identifiers in the new repository (secrets `ROHUB_DEV_*`, variables `ROHUB_DEV_BENCHMARK_IDENTIFIER`/`ROHUB_PROD_BENCHMARK_IDENTIFIER`) and starts its RoHub Integration CI. The GitHub Actions workflow
`create-benchmark-repo.yml` does all of it (secrets `BENCHMARK_REPO_TOKEN`,
`LLM_API_KEY`, `ROHUB_DEV_USERNAME`, `ROHUB_DEV_PASSWORD`). Name repositories after the benchmark's title in its source list,
e.g. `taylor-couette-flow` for "Taylor-Couette Flow" in the DuMux list.

## Edge cases

- **OpenFOAM template cases** (placeholders like `{omega}`) have no values of their
  own: they need a reference description (`reference:`), found automatically in a
  catalog repository.
- **"Not used by any input"** for a reference parameter: that implementation ignores
  it (OpenFOAM rotating-cylinders keeps the outer cylinder fixed) -- fine while every
  configuration uses the value the software has built in.
- **`snakefile-only`** skips the LLM, so parameters have no units and keep their
  software names; check `parameters.json` keys against `run_benchmark.py`.
- **Rate limits**: set `--inference-tpm-budget` well below the account's limit. A
  model's exhausted daily quota switches to another model of the same key, unless
  `--no-model-fallback`.
- **A git URL** is cloned into a temporary folder, so its stored answers are lost
  after the run; `--keep-clone` keeps the clone and its answers.
- **Plain-text prompts instead of the full-screen view**: the full-screen view needs
  a real terminal (stdin and stdout a TTY) and `curses`.

## License

MIT (see `LICENSES/MIT.txt`); the example's DuMux sources are GPL-3.0-or-later.
