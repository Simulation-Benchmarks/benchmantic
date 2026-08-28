# benchmantic

Generates a shareable, machine-readable semantic description (RO-Crate JSON-LD) from a simulation benchmark's source code, and packages a ready-to-run Snakemake workflow around it.

## Pipeline diagram

```
local folder or git URL
  (params.input, C++ source, README, build config)
        │
        ▼
describe_benchmark.py
  discover → infer → review → build → validate
        │
        ├─▶ <name>_benchmark.jsonld
        │     RO-Crate / semantic description
        │     (parameters, metrics, processing steps)
        ├─▶ <name>_dataset.jsonld
        │     dataset provenance (author, publisher,
        │     software dependencies)
        └─▶ Snakefile
              plain file, container-ready

        │
        ├─▶ show_description.py
        │     renders both jsonld files as Markdown tables
        └─▶ verify_description.py
              validates <name>_benchmark.jsonld against semantic_benchmark

workflow.py runs all of the above in one command
```

## Features

- Works from a local checkout or a remote source alike: `<module_dir>` accepts a local folder path, or a GitHub/GitLab/any git-reachable URL, cloned into a throwaway temp directory that's deleted once the run finishes (pass `--keep-clone` to reuse a persistent local clone across runs instead).
- Discovers input parameters and output metrics directly from source (`params.input`, `main.cc`, `problem.hh`), plus license, dependencies, build info, citation, and author details from `README`/`AUTHORS`/`CMakeLists.txt`.
- Uses an LLM (Groq or OpenAI) to infer each parameter's and metric's semantic name, datatype, QUDT unit, and quantity kind — grounded in a compact `getParam<Type>()` call-site excerpt from the C++ source (not the raw config key name alone, and not the whole file). Three layers keep this within a provider's tokens-per-minute budget instead of relying on retries after the fact: requests are estimated and batched below both an item-count cap (`--inference-batch-size`) and a token-size target (`--inference-tpm-budget`), and every request reserves its estimated cost against one shared budget across the whole run so a burst of small requests can't collectively exceed it either.
- Interactive review step: before anything is built into the graph, every inferred parameter/metric is shown in an editable table. Rows below a confidence threshold (default 0.7), or flagged for a structural reason unrelated to the AI's own confidence, get a visual marker — press Enter to accept the table as shown either way, or type a row number to fix one first.
- Cross-benchmark corrections memory: whenever a reviewer edits an AI-inferred value, that correction is remembered and offered back to the LLM as guidance the next time a similarly named/typed parameter shows up, even in a different benchmark module.
- Caches inferred (and reviewed) metadata on disk so repeated runs don't re-query the API, and prunes stale cache entries automatically.
- Assembles a [metadata4ing](https://w3id.org/nfdi4ing/metadata4ing)/RO-Crate 1.1-conformant `@graph` describing the benchmark, its parameter sets, metrics, and license, split across two files: `<benchmark-name>_benchmark.jsonld` (what `semantic_benchmark.BenchmarkLoader` reads) and a sidecar `<benchmark-name>_dataset.jsonld` (author, publisher, software dependencies).
- Generates a plain `Snakefile` from that graph (not zipped), including optional radial-mesh-splitting support for rotating-cylinder-style benchmarks.
- Renders the generated description as human-readable Markdown tables for manual review (merging in the sidecar dataset file automatically), with confidence scores and unit/quantityKind mix-ups flagged.
- Validates the generated benchmark description against the real `semantic_benchmark.BenchmarkLoader`, not a reimplementation of its rules.
- Ships a single `workflow.py` command that runs the whole pipeline and mirrors the verification step's exit code, so it's CI-safe.

## Installation

```bash
git clone https://github.com/Simulation-Benchmarks/benchmantic.git
cd benchmantic
pip install groq rdflib
```

`groq` is the default LLM provider (free tier). To use OpenAI instead, install `openai` and pass `--provider openai` (with `--model`, if you want something other than its default) to `describe_benchmark.py`/`workflow.py`:

```bash
pip install openai
```

`rdflib` is only needed for `verify_description.py`, which uses it via the `semantic_benchmark` package:

```bash
git clone https://github.com/Simulation-Benchmarks/semantic-benchmark.git
```

`verify_description.py` auto-detects a sibling/child `./semantic-benchmark` checkout (or pass `--semantic-benchmark-src`, or set `SEMANTIC_BENCHMARK_SRC`) — no `pip install` of that package needed.

Optionally, install [`roc-validator`](https://github.com/crs4/rocrate-validator) to enable the automatic RO-Crate 1.1 conformance check that runs after generation:

```bash
pip install roc-validator
```

## Requirements

- Python 3.10+
- [Snakemake](https://snakemake.github.io/) and Docker, for running the generated workflow
- An API key for the LLM inference step, set as an environment variable:
  - `GROQ_API_KEY` — default provider, free tier ([console.groq.com/keys](https://console.groq.com/keys))
  - `OPENAI_API_KEY` — used when `--provider openai` is passed

## Quick Start

```bash
python3 workflow.py <module_dir> \
    --scenario-params Cells0,Cells1,Grading0,Radial0,Name,Omega1,Omega2 \
    --full-value-params Radial0 \
    --container-image <docker-image> \
    --container-shared-dir <host-mount-path> \
    --mesh-split --zip-name-flag Problem.Name
```

`<module_dir>` can be the exact benchmark folder, a repository checkout containing one, or a GitHub/GitLab/any git URL (e.g. `https://github.com/org/repo`, or `git@host:org/repo.git`) — in which case it's cloned into a throwaway temp directory that's deleted once this run finishes, so nothing accumulates on disk by default. Pass `--keep-clone` to clone into a persistent local cache instead, reused (fetch + checkout in place) by a later run against the same URL, along with its inference cache — useful if you're iterating on the same remote benchmark repeatedly and don't want to re-query the LLM every time. `--ref <branch-or-tag-or-commit>` pins what's checked out, `--clone-dir` picks an explicit clone location (implies `--keep-clone`), and `--fresh-clone` discards and re-clones an existing persistent one. This walks you through reviewing the AI-inferred parameter/metric metadata, then writes the semantic description, its dataset-provenance sidecar, and a plain `Snakefile` (no zip) to `outputs/<software-name>/`, renders a Markdown review table, and validates the result — exiting non-zero if verification fails. Both jsonld filenames default to a slug of the benchmark's own name, e.g. "rotating cylinders" -> `rotating_cylinders_benchmark.jsonld` + `rotating_cylinders_dataset.jsonld` (neither suffix is doubled up if the name already contains that word) — pass `--benchmark-filename`/`--dataset-filename` to force specific names instead, e.g. if `run_benchmark.py --benchmark-file` is hardcoded to `benchmark.jsonld`.

The review step is one prompt: type a row number to edit it, or press Enter to accept the table as shown and continue — rows below `--review-confidence-threshold` (default `0.7`) or flagged for a structural reason (e.g. a repaired mapping) show a `!`/`?` marker so they're easy to spot first, but Enter always accepts, flagged or not.

For non-interactive/CI runs, add `--skip-review` (and make sure `--scenario-params` is always passed, since omitting it also triggers an interactive prompt). Skip individual pipeline steps with `--skip-show` / `--skip-check`. Run any script with `--help` for its full option list.

## Architecture / Pipeline walkthrough

`describe_benchmark.py` is the single entry point, and runs five stages in process (calling `metadata.builder` and `snakefile.generator` directly, not via subprocess):

1. **Discover** (`metadata.repo_source`, `metadata.repository`, `metadata.parameters`, `metadata.metrics`) — first resolves `module_dir`: a local path is used as-is; a git URL is cloned into a throwaway temp directory, deleted automatically once the run finishes (or, with `--keep-clone`, into a persistent cache directory instead — default `.benchmantic_repo_cache/`, override with `BENCHMANTIC_REPO_CACHE` or `--clone-dir`). Then scans the resolved module for parameter and metric candidates, and scrapes license, dependency, build, citation, and author information from source and README. Parameters can be selected via `--scenario-params`, or interactively if omitted.
2. **Infer** (`ai.cache`, `ai.inference`) — for every candidate not already cached, asks an LLM to infer its physical meaning. This is broken down further inside the `ai/` package:
   - `ai.prompts` builds the prompt from the benchmark's description, the candidate parameters/metrics, and any relevant prior human corrections (see `ai.corrections` below) — deliberately *not* the full `main.cc`/`problem.hh` source. Each parameter already carries its own `getParam<>()` call-site excerpt (`metadata.parameters.attach_cpp_hints`) and each metric its own surrounding-code snippet (`metadata.metrics.discover_metrics`), which is far smaller and just as code-grounded; the full file is only searched as a last-resort fallback for the rare item with no per-item context of its own.
   - `ai.inference` queries the configured provider (Groq by default, or OpenAI), retrying up to 3 times on failure, and failing fast (no pointless retries/waits) on an unrecoverable size/rate-limit response instead of exhausting all 3 attempts on something retrying can't fix. Before any of that: more than `--inference-batch-size` (default 8) not-yet-cached items are split into multiple independent requests, each chunk is split *again* if its own estimated token cost (prompt + expected completion) exceeds `--inference-tpm-budget` (default 9,000) regardless of item count, and every request -- across the whole run, not just within one batch -- reserves its estimated cost against one shared tokens-per-minute ledger before sending, sleeping if needed so the run's real request rate stays under that budget too.
   - `ai.validation` checks the response for missing fields, out-of-range indices, and unit/quantityKind mix-ups, correcting or flagging them.
   - `ai.cache` persists the final (post-review) result beside the module, so each item is inferred only once.
3. **Review** (`ai.review`) — prints every parameter/metric in an editable table (semantic name, datatype, unit, quantity kind, confidence). One prompt: type a row number to edit any field, or press Enter to accept the table as shown and continue. Rows below the confidence threshold get a `!`, and rows with a structural (not confidence) issue -- e.g. the model omitted the `ini`/`key` field and it had to be reconstructed positionally, see `ai.validation` -- get a `?`; both are just visual flags, Enter always accepts regardless. Skipped entirely with `--skip-review`. Every value actually changed here is recorded by `ai.corrections`, which persists it (matched by exact or fuzzy key similarity) so a similarly named/typed parameter in a *different* benchmark benefits from it too, next time it's inferred.
4. **Build** (`metadata.builder`) — wraps each reviewed value, along with the scraped manifest info, into typed nodes inside one metadata4ing `@graph` (`GraphBuilder`). Author, publisher, and software dependencies are assembled into a *separate* graph here too (`GraphBuilder.build_dataset_graph()`) rather than merged into the same one -- see the next step.
5. **Validate & Generate** (`metadata.builder`, `snakefile.generator`, `snakefile.renderer`) — checks the benchmark graph against the RO-Crate 1.1 profile, then writes `<benchmark-name>_benchmark.jsonld` (parameters, metrics, processing steps, RO-Crate root), a sidecar `<benchmark-name>_dataset.jsonld` (author, publisher, software dependencies, linked back to the benchmark file via `schema:isPartOf`), and a plain `Snakefile` (not zipped).

Two additional scripts sit downstream of generation:

- **`show_description.py`** loads the benchmark file, auto-discovers its sibling `_dataset.jsonld` file next to it (or takes `--dataset-jsonld` explicitly), and renders both together as Markdown tables (manifest info, dependencies, parameters, metrics) for human review, flagging any unit/quantityKind mix-ups. Unlike the interactive review step, this reflects the fully built graph — including manifest fields (license, authors, dependencies) and resolved case values that review never touches — and is saved to disk (`review.md`) as a durable record.
- **`verify_description.py`** loads the benchmark file (not the dataset sidecar -- it has no bearing on the check) with the real `semantic_benchmark.BenchmarkLoader` and checks that every processing step, parameter set, and metric field mapping actually resolves — not just that the file is valid JSON-LD.

To fix a value after the fact (e.g. a run done with `--skip-review`, or something spotted later in `review.md`): edit the relevant entry in `.parameter_metadata_cache.json` (or `.metric_metadata_cache.json`) inside the module directory, then re-run `describe_benchmark.py` **without** `--clear-cache` so the fix is reused rather than re-inferred, and re-run `verify_description.py` to confirm.

## Repository structure

```
benchmantic/
├── describe_benchmark.py    # entry point: orchestrates metadata + Snakefile generation
├── show_description.py       # renders the benchmark + dataset jsonld files as Markdown for review
├── verify_description.py     # validates the benchmark jsonld file against semantic_benchmark
├── workflow.py                # runs the three scripts above as one CI-friendly pipeline
├── utils.py                   # shared string/number/file helpers
├── metadata/                  # benchmark.jsonld generation
│   ├── builder.py             #   orchestrates manifest + RO-Crate @graph construction (benchmark + dataset docs)
│   ├── graph.py               #   shared @id-resolution / node-lookup helpers
│   ├── repo_source.py         #   resolves module_dir: local path, or clone/update a git URL
│   ├── repository.py          #   repo scanning: README/AUTHORS/SPDX/CMake discovery
│   ├── parameters.py          #   parameter discovery + case resolution
│   ├── metrics.py             #   output/solution metric discovery
│   ├── publication.py         #   literature citation extraction
│   └── software.py            #   simulation-software detection (DuMux, OpenFOAM, ...)
├── ai/                        # LLM-based semantic metadata inference
│   ├── prompts.py              #   prompt templates for parameters and metrics
│   ├── inference.py            #   provider config + request/retry loop
│   ├── validation.py           #   response validation and repair
│   ├── review.py                #   interactive CLI review/edit step with confidence gating
│   ├── cache.py                 #   on-disk inference caching (per module)
│   └── corrections.py           #   cross-benchmark human-correction memory
└── snakefile/                  # executable-workflow generation
    ├── generator.py             #   derives Snakefile inputs from benchmark.jsonld
    └── renderer.py               #   renders the actual Snakefile text
```

## License

MIT — see [`LICENSES/MIT.txt`](LICENSES/MIT.txt). Licensing follows the [REUSE](https://reuse.software) convention; run `reuse lint` to verify compliance.
