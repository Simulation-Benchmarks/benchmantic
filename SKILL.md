---
name: benchmantic
description: Generates a shareable, machine-readable semantic description (RO-Crate JSON-LD) of a simulation-model benchmark from its source code, using an LLM to infer parameter/metric units and quantity kinds, then packages a ready-to-run Snakemake workflow. Use when a user wants to turn a simulation benchmark's source code (params.input, main.cc, problem.hh) into a benchmark.jsonld file, review or verify such a description, or build a container-ready Snakefile workflow for a CFD/simulation benchmark.
license: MIT
compatibility: Requires Python 3.10+, Snakemake, and Docker (for container-based benchmark execution). Requires a GROQ_API_KEY (default LLM provider, free tier) or, with --provider openai, an OPENAI_API_KEY. Network access needed to query the LLM provider and pull container images.
metadata:
  repository: https://github.com/Simulation-Benchmarks/benchmantic
  maintainer: Simulation-Benchmarks
---

# benchmantic

benchmantic turns a simulation benchmark's source code into a standardized, machine-readable semantic description (`benchmark.jsonld`, RO-Crate 1.1 / metadata4ing), and packages a ready-to-run Snakemake workflow around it. It's used to prepare and validate benchmark entries for the NFDI4Ing Model Validation Platform and similar consumers.

## When to use this skill

Use this skill when the task involves any of:

- Extracting a semantic description (`benchmark.jsonld`) from a simulation model's source code or repository.
- Inferring units, quantity kinds, or semantic names for simulation parameters/metrics from source code context.
- Rendering an existing `benchmark.jsonld` as a human-readable Markdown table for review.
- Verifying that a generated description conforms to the `semantic_benchmark` schema/loader.
- Producing a packaged, container-runnable Snakefile workflow, including mesh-splitting options for rotating-cylinder-style benchmarks.
- Running the full describe → show → verify pipeline end-to-end, e.g. for CI.

## Installation

```bash
git clone https://github.com/Simulation-Benchmarks/benchmantic.git
cd benchmantic
pip install groq rdflib
```

`groq` is the default LLM provider (free tier). To use OpenAI instead, `pip install openai` and pass `--provider openai` (plus `--model`, optionally). `rdflib` is needed for `verify_description.py`, which also needs a `semantic-benchmark` checkout:

```bash
git clone https://github.com/Simulation-Benchmarks/semantic-benchmark.git
```

(auto-detected as a sibling/child directory, or pass `--semantic-benchmark-src` / set `SEMANTIC_BENCHMARK_SRC`).

## Quickstart

Run the full pipeline (generate, review, verify) in one command:

```bash
python3 workflow.py <module_dir> \
    --scenario-params Cells0,Cells1,Grading0,Radial0,Name,Omega1,Omega2 \
    --full-value-params Radial0 \
    --container-image <docker-image> \
    --container-shared-dir <host-mount-path> \
    --mesh-split --zip-name-flag Problem.Name
```

`<module_dir>` can be the exact benchmark folder or a repository checkout containing one. This writes `benchmark.jsonld` and a packaged `Snakefile` to `outputs/<software-name>/`, renders a Markdown review table, and validates the result — exiting non-zero if verification fails. Skip individual steps with `--skip-show` / `--skip-check`.

## Step-by-step

`describe_benchmark.py` is the entry point and runs four stages in process:

1. **Discover** (`metadata.repository`, `metadata.parameters`, `metadata.metrics`) — scans `params.input`, `main.cc`, `problem.hh`, `README`, and `CMakeLists.txt` for parameter/metric candidates, license, dependencies, build info, citation, and authors. Pass `--scenario-params` to select case-varying parameters non-interactively (required for unattended/CI runs — omitting it triggers an interactive prompt).
2. **Infer** (`ai.cache`, `ai.inference`) — for candidates not already cached, asks the configured LLM to infer each one's semantic name, datatype, QUDT unit, and quantity kind, using the source code and `getParam<Type>()` call-site hints as context. Retries up to 3 times before failing (or falling back to a low-confidence placeholder with `--fallback-on-error`).
3. **Build** (`metadata.builder`) — wraps each value plus the scraped manifest info into typed nodes inside one metadata4ing `@graph`.
4. **Validate & Generate** (`metadata.builder`, `snakefile.generator`, `snakefile.renderer`) — checks the graph against the RO-Crate 1.1 profile, then writes `benchmark.jsonld` and a packaged `Snakefile` to `outputs/<software-name>/`.

Review and verify the result:

- `show_description.py benchmark.jsonld` renders it as Markdown tables, flagging low-confidence inferences and unit/quantityKind mix-ups.
- `verify_description.py benchmark.jsonld` loads it with the real `semantic_benchmark.BenchmarkLoader` and confirms every parameter set and metric field mapping resolves.

To fix a wrong inferred value: run `show_description.py`, edit the relevant entry in `.parameter_metadata_cache.json` (or `.metric_metadata_cache.json`) inside the module directory, re-run `describe_benchmark.py` **without** `--clear-cache` so the fix is reused, then re-run `verify_description.py` to confirm.

## Repository layout

- `describe_benchmark.py` — entry point; orchestrates metadata + Snakefile generation.
- `show_description.py` — renders `benchmark.jsonld` as Markdown.
- `verify_description.py` — validates `benchmark.jsonld` against `semantic_benchmark`.
- `workflow.py` — runs the three scripts above as one CI-friendly pipeline.
- `utils.py` — shared string/number/file helpers.
- `metadata/` — `builder.py`, `graph.py`, `repository.py`, `parameters.py`, `metrics.py`, `publication.py`, `software.py` — benchmark.jsonld generation.
- `ai/` — `prompts.py`, `inference.py`, `validation.py`, `cache.py` — LLM-based semantic metadata inference.
- `snakefile/` — `generator.py`, `renderer.py` — executable-workflow generation.

## Common edge cases

- **No `--scenario-params` supplied**: falls back to an interactive prompt; unattended/CI runs must pass this flag explicitly or the run will hang waiting for input.
- **Missing required Snakefile flags**: `--container-image` and `--container-shared-dir` are required even if you only care about the `benchmark.jsonld` output.
- **Verification failures**: a failing `verify_description.py` run should block downstream packaging/publishing — `workflow.py` already propagates this as a non-zero exit code.
- **Rotating-cylinder / radial-mesh benchmarks**: only enable `--mesh-split` (and its `--radial-cells-flag`/`--angular-cells-flag`/`--grading-flag`) when the benchmark actually uses a radial mesh; it's not a general-purpose flag.
- **Missing API key**: `GROQ_API_KEY` (or `OPENAI_API_KEY` with `--provider openai`) must be set before the inference step runs.

## License

MIT — see `LICENSES/MIT.txt`. Licensing follows the [REUSE](https://reuse.software) convention.
