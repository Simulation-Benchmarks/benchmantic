# benchmantic

Generates a shareable, machine-readable semantic description (RO-Crate JSON-LD) from a simulation benchmark's source code, and packages a ready-to-run Snakemake workflow around it.

## Pipeline diagram

```
                        ┌─────────────────────┐
 source code folder ──▶ │ describe_benchmark.py│──▶ benchmark.jsonld  (RO-Crate / semantic description)
 (params, C++, README,  │  discover → infer →  │──▶ Snakefile          (packaged, container-ready)
  build config)         │  build → validate     │
                        └─────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           show_description.py             verify_description.py
         (Markdown review tables)      (validates against semantic_benchmark)

              workflow.py runs all of the above in one command
```

## Features

- Discovers input parameters and output metrics directly from source (`params.input`, `main.cc`, `problem.hh`), plus license, dependencies, build info, citation, and author details from `README`/`AUTHORS`/`CMakeLists.txt`.
- Uses an LLM (Groq or OpenAI) to infer each parameter's and metric's semantic name, datatype, QUDT unit, and quantity kind — grounded in `getParam<Type>()` call-site hints from the C++ source, not just the raw config key name.
- Caches inferred metadata on disk so repeated runs don't re-query the API, and prunes stale cache entries automatically.
- Assembles a [metadata4ing](https://w3id.org/nfdi4ing/metadata4ing)/RO-Crate 1.1-conformant `@graph` describing the benchmark, its parameter sets, metrics, license, software, and authors.
- Generates a packaged `Snakefile` from that graph, including optional radial-mesh-splitting support for rotating-cylinder-style benchmarks.
- Renders the generated description as human-readable Markdown tables for manual review, with confidence scores and unit/quantityKind mix-ups flagged.
- Validates the generated description against the real `semantic_benchmark.BenchmarkLoader`, not a reimplementation of its rules.
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

`<module_dir>` can be the exact benchmark folder or a repository checkout containing one. This writes `benchmark.jsonld` and a packaged `Snakefile` to `outputs/<software-name>/`, renders a Markdown review table, and validates the result — exiting non-zero if verification fails. Skip individual steps with `--skip-show` / `--skip-check`. Run any script with `--help` for its full option list.

## Architecture / Pipeline walkthrough

`describe_benchmark.py` is the single entry point, and runs four stages in process (calling `metadata.builder` and `snakefile.generator` directly, not via subprocess):

1. **Discover** (`metadata.repository`, `metadata.parameters`, `metadata.metrics`) — scans the module for parameter and metric candidates, and scrapes license, dependency, build, citation, and author information from source and README. Parameters can be selected via `--scenario-params`, or interactively if omitted.
2. **Infer** (`ai.cache`, `ai.inference`) — for every candidate not already cached, asks an LLM to infer its physical meaning. This is broken down further inside the `ai/` package:
   - `ai.prompts` builds the prompt, assembling the benchmark's description, the candidate parameters/metrics, and the relevant source code as context.
   - `ai.inference` queries the configured provider (Groq by default, or OpenAI), retrying up to 3 times on failure.
   - `ai.validation` checks the response for missing fields, out-of-range indices, and unit/quantityKind mix-ups, correcting or flagging them.
   - `ai.cache` persists the result beside the module, so each item is inferred only once.
3. **Build** (`metadata.builder`) — wraps each inferred value, along with the scraped manifest info, into typed nodes inside one metadata4ing `@graph` (`GraphBuilder`).
4. **Validate & Generate** (`metadata.builder`, `snakefile.generator`, `snakefile.renderer`) — checks the graph against the RO-Crate 1.1 profile, then writes `benchmark.jsonld` and a packaged `Snakefile`.

Two additional scripts sit downstream of generation:

- **`show_description.py`** loads `benchmark.jsonld` and renders it as Markdown tables (manifest info, dependencies, parameters, metrics) for human review, flagging any unit/quantityKind mix-ups.
- **`verify_description.py`** loads `benchmark.jsonld` with the real `semantic_benchmark.BenchmarkLoader` and checks that every processing step, parameter set, and metric field mapping actually resolves — not just that the file is valid JSON-LD.

To correct a wrong inferred value: run `show_description.py` to spot the issue, edit the relevant entry in `.parameter_metadata_cache.json` (or `.metric_metadata_cache.json`) inside the module directory, then re-run `describe_benchmark.py` **without** `--clear-cache` so the fix is reused rather than re-inferred, and re-run `verify_description.py` to confirm.

## Repository structure

```
benchmantic/
├── describe_benchmark.py    # entry point: orchestrates metadata + Snakefile generation
├── show_description.py      # renders benchmark.jsonld as Markdown for review
├── verify_description.py    # validates benchmark.jsonld against semantic_benchmark
├── workflow.py               # runs the three scripts above as one CI-friendly pipeline
├── utils.py                  # shared string/number/file helpers
├── metadata/                 # benchmark.jsonld generation
│   ├── builder.py            #   orchestrates manifest + RO-Crate @graph construction
│   ├── graph.py              #   shared @id-resolution / node-lookup helpers
│   ├── repository.py         #   repo scanning: README/AUTHORS/SPDX/CMake discovery
│   ├── parameters.py         #   parameter discovery + case resolution
│   ├── metrics.py            #   output/solution metric discovery
│   ├── publication.py        #   literature citation extraction
│   └── software.py           #   simulation-software detection (DuMux, OpenFOAM, ...)
├── ai/                       # LLM-based semantic metadata inference
│   ├── prompts.py            #   prompt templates for parameters and metrics
│   ├── inference.py          #   provider config + request/retry loop
│   ├── validation.py         #   response validation and repair
│   └── cache.py              #   on-disk inference caching
└── snakefile/                # executable-workflow generation
    ├── generator.py          #   derives Snakefile inputs from benchmark.jsonld
    └── renderer.py            #   renders the actual Snakefile text
```

## License

MIT — see [`LICENSES/MIT.txt`](LICENSES/MIT.txt). Licensing follows the [REUSE](https://reuse.software) convention; run `reuse lint` to verify compliance.
