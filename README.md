# benchmantic

benchmantic reads the source code of a simulation benchmark and writes a
machine-readable description of it (JSON-LD: parameters, configurations,
metrics, with units), a Snakemake workflow that runs it in a container, and,
if you like, a whole benchmark repository like
[rotating-cylinders](https://github.com/Simulation-Benchmarks/rotating-cylinders)
-- on GitHub, with its CI.

An LLM works out what each input and output means (name, unit, quantity kind);
you review its answers, and they are stored, so a benchmark is described the
same way every time.

```
implementation (folder or GitHub/GitLab link)
   │
   ├─ workflow.py ──────────── describe ─▶ outputs/<software>/
   │                           verify       <name>_benchmark.jsonld   the description
   │                                        <name>_dataset.jsonld     authors, license, dependencies
   │                                        <name>_mapping.json       parameters -> software inputs
   │                                        Snakefile                 runs every configuration
   │                                        review.md                 everything as tables
   │
   └─ create_benchmark_repo.py ──────────▶ a benchmark repository (optionally on GitHub)
```

## Install

```bash
git clone https://github.com/Simulation-Benchmarks/benchmantic.git
cd benchmantic
pip install -r requirements.txt     # openai, pyyaml, rdflib, semantic-benchmark
pip install reuse                   # for create_benchmark_repo.py
pip install roc-validator           # optional: RO-Crate 1.1 check of the description
```

Python 3.10 or newer. Running the generated workflow needs Snakemake and
Apptainer (or Docker).

## Quick start

Try the example -- Taylor-Couette flow (rotating cylinders) with DuMux, from the
[DuMux benchmark list](https://dumux.org/docs/doxygen/master/benchmarks.html):

```bash
python3 workflow.py --config examples/taylor-couette-flow/benchmantic.yaml
python3 create_benchmark_repo.py --config examples/taylor-couette-flow/benchmantic.yaml
```

The first command writes `outputs/dumux/` and checks the description; the
second writes the repository to `outputs/repositories/taylor-couette-flow/`.
The example's reviewed LLM answers are stored with it, so this needs no LLM key.

For your own benchmark, write a run config:

```yaml
implementation: /path/to/your/benchmark   # a folder, or a GitHub/GitLab link
software: DuMux                           # the simulation software
container: registry.example.org/my-benchmark:1.0   # the image it runs in
```

and run `python3 workflow.py --config my-benchmark.yaml`. benchmantic shows
the inputs it found and asks which ones differ between configurations, asks
the LLM what they mean, and shows you its answers to review.

## The run config

The same keys work for every simulation software
([`run-config-template.yaml`](run-config-template.yaml) explains each one):

| Key | Meaning |
|---|---|
| `implementation` | your code: a folder or file, a git URL, or a link to a folder or file in a repository (required) |
| `software` | the simulation software, e.g. `DuMux` (default: the LLM works it out) |
| `container` | the container image the simulation runs in (required) |
| `reference` | an existing description of the same benchmark to map onto; `none` for none (default: looked for) |
| `llm_provider`, `llm_model` | which LLM -- see below |
| `review_below_confidence` | you review every answer less sure than this (default 0.7) |
| `verbose` | show more detail while running |
| `software_description` | your own description of the software (see below) |

Every command-line option also works as a key, by its name
(`python3 describe_benchmark.py --help` lists them), e.g. `scenario-params`
(the inputs that differ between configurations) and `skip-review` (for runs
without a terminal). A value on the command line wins over the file.

## The LLM

| `llm_provider` | Key | Default model |
|---|---|---|
| `groq` | `GROQ_API_KEY` ([free tier](https://console.groq.com/keys)) | `openai/gpt-oss-120b` |
| `claude` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` |
| `claude-code` | none: your Claude plan, through [Claude Code](https://code.claude.com/docs/en/setup) | `sonnet` |
| `codex` | none: your ChatGPT plan, through [Codex](https://developers.openai.com/codex/quickstart) | Codex's default |

The key comes from `LLM_API_KEY` or the variable above, `--llm-api-key`, or
is asked for -- never from the run config, which gets shared. Without
`llm_provider`, the first LLM whose key is set is used.

- **Review.** Answers below `review_below_confidence`, or with a problem
  benchmantic spotted, are shown for review (a full-screen view on a terminal,
  plain text otherwise); `skip-review: true` accepts them as they are.
- **Stored answers.** The reviewed answers are saved next to the
  implementation (`.parameter_metadata_cache.json`,
  `.metric_metadata_cache.json`) and reused; `--clear-cache` asks again. To fix
  an answer later, edit the file and run again.
- **Corrections** you make are remembered and shown to the LLM for similar
  inputs of other benchmarks.
- **Limits.** Requests are batched under a tokens-per-minute budget
  (`--inference-batch-size`, `--inference-tpm-budget`); when a model's daily
  quota runs out, another model of the same key is tried
  (`--no-model-fallback` stops instead).

## Supported software

benchmantic holds no code for any particular software. A short *software
description* (`software_descriptions/<name>.yaml`) says which files make up an
implementation, where its configurations, inputs and results are, and how one
configuration is run; [`software/template.yaml`](software/template.yaml)
documents every key. Three come with benchmantic:

| Software | Recognised by | Inputs | Metrics |
|---|---|---|---|
| DuMux | `main.cc` + `problem.hh` | `params.input` entries, with their `getParam<T>()` call sites | JSON keys written in `main.cc` |
| OpenFOAM | a case with `system/controlDict`, or a zip with one | dictionary entries; template placeholders `{name}` | keys of the dict a Python script writes to `solution_metrics.json` |
| FEniCSx | the script or notebook the link points at | numbers in the code (`r = 0.05`) | values of the code's variables when it ends (`max(C_D)`) |

For any other software the LLM writes the description from your repository,
tries it, and corrects it until benchmantic can read the implementation with
it; it is saved as `.benchmantic_software.yaml` next to the implementation
(and, if you agree, in `software_descriptions/`). Write the part you know
yourself with `software_description: my-code.yaml` -- the LLM fills in the rest.

Configurations can also be one parameter file each (e.g. `parameters_05.json`:
`cases.files` in the description).

## Parameter names across software

Parameters get software-neutral names (`angular_velocity_inner_cylinder`), and
`<name>_mapping.json` links them to the software's own inputs. When a
*reference* description of the same benchmark exists -- `reference:` in the run
config, a description in the repository (e.g.
`benchmark/1.0.0/minimal-configurations.json`), or benchmantic's own output for
another software -- the LLM maps the software's inputs onto the reference's
parameters, as one expression per input, which you review:

| Input | Expression |
|---|---|
| DuMux `Grid.Cells0` | `{cells_radial // 2} {cells_radial // 2}` |
| DuMux `Grid.Radial0` | `{inner_radius} {(inner_radius + outer_radius) / 2} {outer_radius}` |
| OpenFOAM `geom/ntheta` | `{cells_angular // 4}` |

The expressions are checked against every configuration, and a unit mismatch
(degrees vs radians) is flagged. So describing DuMux first and OpenFOAM second
gives both the same parameters. `--parameter-names software` keeps the
software's own names instead.

The research problem and mathematical model a benchmark is linked to are
looked up on the [MaRDI portal](https://portal.mardi4nfdi.de)
(`--no-mardi-lookup` to skip; `--research-problem-id`, `--math-model-id` to set
them).

## What it writes

`--outputs` (or the choice benchmantic offers on the first run) picks the files:

| Preset | Description + review | Dataset | Snakefile |
|---|:-:|:-:|:-:|
| `standard` (default) | ✓ | ✓ | ✓ |
| `snakefile` | ✓ | | ✓ |
| `dataset` | ✓ | ✓ | |
| `description-only` | ✓ | | |
| `snakefile-only` | | | ✓ (no LLM; parameters keep their software names and no units) |

`workflow.py` then renders the description as Markdown (`show_description.py`
-> `review.md`) and checks it (`verify_description.py`: the SHACL shapes of
[semantic-benchmark](https://pypi.org/project/semantic-benchmark/), then
loading it with its `BenchmarkLoader`); its exit code is the check's, so it can
gate CI. The Snakefile reads each configuration's `parameters.json` (as the
catalog's `run_benchmark.py` writes it) and runs the simulation in the
container; set `BENCHMARK_CONTAINER` to a local `.sif` to use that instead of
the published image.

`run_configs.py` runs `workflow.py` for every run config in a folder.

## A benchmark repository

`create_benchmark_repo.py` turns the outputs into a repository laid out like
[rotating-cylinders](https://github.com/Simulation-Benchmarks/rotating-cylinders):
README, CITATION.cff, codemeta.json, REUSE licensing,
`benchmark/<version>/minimal-configurations.json`, `<software>/Snakefile` and
`run_benchmark.py`, the container's folder, and GitHub Actions that build the
container, run every configuration and check its metrics. The files are
[`repository_template/`](repository_template/) with the benchmark filled in.
It reads the `repository:` block of the run config:

```yaml
repository:
  name: taylor-couette-flow            # the benchmark's title in the DuMux list
  title: Taylor-Couette Flow Benchmark
  summary: A benchmark for ...
  documentation: documentation.md      # becomes docs/benchmark-documentation.md
  container: container                 # the Dockerfile's folder (or a GitHub/GitLab link)
  authors: ["Given Family; ORCID; affiliation"]
```

```bash
python3 create_benchmark_repo.py --config my-benchmark.yaml --org Sarbani-Roy --github --watch
```

`--github` creates the repository with the GitHub CLI (`gh`) and pushes it --
or adds a commit to it, if it exists; `--watch` waits for its Actions and fails
unless they pass.

## GitHub Actions

| Workflow | When | What |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | every push and pull request | tests and REUSE lint; the example described and turned into a repository |
| [`create-benchmark-repo.yml`](.github/workflows/create-benchmark-repo.yml) | Actions tab -> *Run workflow* | describes a benchmark (input: its run config), creates its repository in a GitHub account and waits for that repository's Actions |

*Create benchmark repository* needs two repository secrets: `BENCHMARK_REPO_TOKEN`,
a personal access token of the target account (classic: scopes `repo` and
`workflow` -- the built-in token cannot create repositories), and
`LLM_API_KEY` (provider: the repository variable `LLM_PROVIDER`, default
`groq`). Its input *ask_llm* asks the LLM again for everything; otherwise only
for what has no stored answer -- everything, for a new benchmark.

Locally: `pip install -r requirements.txt pytest numpy httpx reuse`, then
`python -m pytest` and `reuse lint`.

## Layout

```
workflow.py              describe -> show -> verify, in one command
describe_benchmark.py    describe: the description, dataset, mapping and Snakefile
show_description.py      a description as Markdown tables (review.md)
verify_description.py    checks a description against semantic-benchmark
create_benchmark_repo.py a benchmark repository from the outputs
run_configs.py           workflow.py for every run config in a folder
config.py                run configs
metadata/                reading the implementation, building the JSON-LD
ai/                      the LLM: prompts, requests, checks, review, stored answers
software/                reading any software from its description
software_descriptions/   DuMux, OpenFOAM, FEniCSx
snakefile/               writing the Snakefile
repository_template/     the files of a benchmark repository
examples/                Taylor-Couette flow with DuMux
tests/
```

## License

MIT -- see [`LICENSES/MIT.txt`](LICENSES/MIT.txt). The example's DuMux sources
are GPL-3.0-or-later. Licensing follows [REUSE](https://reuse.software); run
`reuse lint` to check.
