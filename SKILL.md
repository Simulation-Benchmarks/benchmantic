---
name: benchmantic
description: Generates a shareable, machine-readable semantic description (RO-Crate JSON-LD) of a simulation-model benchmark from its source code, using an LLM plus an interactive human review step to infer parameter/metric units and quantity kinds, then packages a ready-to-run Snakemake workflow. Use when a user wants to turn a simulation benchmark's source code (params.input, main.cc, problem.hh) into a benchmark.jsonld file, review or verify such a description, or build a container-ready Snakefile workflow for a CFD/simulation benchmark.
license: MIT
compatibility: Requires Python 3.10+, Snakemake, and Docker (for container-based benchmark execution). Requires a GROQ_API_KEY (default LLM provider, free tier) or, with --provider openai, an OPENAI_API_KEY. The interactive review step needs a real terminal (input()) unless run with --skip-review. Network access needed to query the LLM provider and pull container images.
metadata:
  repository: https://github.com/Simulation-Benchmarks/benchmantic
  maintainer: Simulation-Benchmarks
---

# benchmantic

benchmantic turns a simulation benchmark's source code into a standardized, machine-readable semantic description (`benchmark.jsonld`, RO-Crate 1.1 / metadata4ing), and packages a ready-to-run Snakemake workflow around it. It's used to prepare and validate benchmark entries for the NFDI4Ing Model Validation Platform and similar consumers.

Pipeline: **discover → infer → review → build → validate**. AI inference proposes parameter/metric metadata; a human reviews and can edit any field before it's built into the graph; corrections are remembered across benchmarks and fed back into future prompts.

## When to use this skill

Use this skill when the task involves any of:

- Extracting a semantic description (`benchmark.jsonld`) from a simulation model's source code or repository.
- Inferring units, quantity kinds, or semantic names for simulation parameters/metrics from source code context.
- Reviewing/editing AI-inferred metadata before it's committed to a benchmark description.
- Rendering an existing `benchmark.jsonld` as a human-readable Markdown table for review.
- Verifying that a generated description conforms to the `semantic_benchmark` schema/loader.
- Producing a packaged, container-runnable Snakefile workflow, including mesh-splitting options for rotating-cylinder-style benchmarks.
- Running the full describe → review → build → validate pipeline end-to-end, e.g. for CI (with `--skip-review`).

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

`<module_dir>` can be the exact benchmark folder, a repository checkout containing one, or a GitHub/GitLab/any git URL (e.g. `https://github.com/org/repo`, `git@host:org/repo.git`) — cloned into a throwaway temp directory deleted once the run finishes, by default; pass `--keep-clone` to clone into a persistent local cache reused across runs instead (along with its inference cache), `--ref` to pin a branch/tag/commit, `--clone-dir` to pick an explicit clone location (implies `--keep-clone`), or `--fresh-clone` to discard and re-clone an existing persistent one. This walks you through reviewing the AI-inferred metadata (edit any field, or press Enter to accept the table as shown), writes the semantic description, its dataset-provenance sidecar, and a plain `Snakefile` (no zip) to `outputs/<software-name>/`, renders a Markdown review table, and validates the result — exiting non-zero if verification fails. Both jsonld filenames default to a slug of the benchmark's own name -- `<benchmark-name>_benchmark.jsonld` (parameters/metrics/processing steps) + `<benchmark-name>_dataset.jsonld` (author/publisher/dependencies), neither suffix doubled up if the name already contains that word -- pass `--benchmark-filename`/`--dataset-filename` to force specific names instead. For CI/non-interactive use, add `--skip-review` (this step needs a real terminal). Skip individual pipeline steps with `--skip-show` / `--skip-check`.

## Step-by-step

`describe_benchmark.py` is the entry point and runs five stages in process:

1. **Discover** (`metadata.repo_source`, `metadata.repository`, `metadata.parameters`, `metadata.metrics`) — resolves `module_dir` first: used as-is if it's a local path; a git URL is cloned into a throwaway temp directory, deleted automatically once the run finishes (or, with `--keep-clone`, into a persistent cache instead — default `.benchmantic_repo_cache/`, override with `BENCHMANTIC_REPO_CACHE` or `--clone-dir`). Then scans `params.input`, `main.cc`, `problem.hh`, `README`, and `CMakeLists.txt` for parameter/metric candidates, license, dependencies, build info, citation, and authors. Pass `--scenario-params` to select case-varying parameters non-interactively (required for unattended/CI runs — omitting it triggers an interactive prompt).
2. **Infer** (`ai.cache`, `ai.inference`) — for candidates not already cached, asks the configured LLM to infer each one's semantic name, datatype, QUDT unit, and quantity kind, using each parameter's own `getParam<Type>()` call-site excerpt (or each metric's own surrounding-code snippet) plus any relevant prior human corrections (`ai.corrections`) as context -- not the full `main.cc`/`problem.hh` source, which used to dominate request size and risk hitting a provider's per-request token/payload limit. Rate-limit safety is layered rather than reactive-only: more than `--inference-batch-size` (default 8) not-yet-cached items are split into multiple independent requests; each resulting chunk is split *again* if its own estimated token cost exceeds `--inference-tpm-budget` (default 9,000), regardless of item count; and every request made during the run reserves its estimated cost against one shared tokens-per-minute budget before sending, so individually-small requests can't collectively exceed it either. Retries up to 3 times before failing (or falling back to a low-confidence placeholder with `--fallback-on-error`) -- except an unrecoverable size/rate-limit response, which fails immediately instead of wasting retries on something that can't succeed.
3. **Review** (`ai.review`) — prints a table of every parameter/metric (name, datatype, unit, quantity kind, confidence). One prompt: type a row number to edit any field, or press Enter to accept the table as shown and continue. Rows below `--review-confidence-threshold` (default `0.7`) get a `!`, and rows needing a structural check (e.g. a model-omitted `ini`/`key` field reconstructed positionally) get a `?` -- both are just visual flags; Enter always accepts either way. Skipped entirely with `--skip-review`. Whatever a reviewer changes is recorded by `ai.corrections` and offered back to the LLM as guidance for similarly named/typed items in *other* benchmarks.
4. **Build** (`metadata.builder`) — wraps each reviewed value plus the scraped manifest info into typed nodes inside one metadata4ing `@graph`. Author, publisher, and software dependencies are assembled into a *separate* graph here (`GraphBuilder.build_dataset_graph()`), not merged into the same one.
5. **Validate & Generate** (`metadata.builder`, `snakefile.generator`, `snakefile.renderer`) — checks the benchmark graph against the RO-Crate 1.1 profile, then writes `<benchmark-name>_benchmark.jsonld`, a sidecar `<benchmark-name>_dataset.jsonld` (linked back via `schema:isPartOf`), and a plain `Snakefile` to `outputs/<software-name>/`.

Review and verify the result downstream of generation:

- `show_description.py <benchmark-name>_benchmark.jsonld` auto-discovers the sibling `_dataset.jsonld` file next to it and renders both together as Markdown tables — including manifest fields (license, authors, dependencies) and resolved case values the interactive review step doesn't cover — saved to `review.md` as a durable record.
- `verify_description.py <benchmark-name>_benchmark.jsonld` loads it with the real `semantic_benchmark.BenchmarkLoader` and confirms every parameter set and metric field mapping resolves (the dataset sidecar isn't involved in this check).

To fix a value after the fact (e.g. a `--skip-review` run, or something spotted later in `review.md`): edit the relevant entry in `.parameter_metadata_cache.json` (or `.metric_metadata_cache.json`) inside the module directory, re-run `describe_benchmark.py` **without** `--clear-cache` so the fix is reused, then re-run `verify_description.py` to confirm.

## Repository layout

- `describe_benchmark.py` — entry point; orchestrates metadata + Snakefile generation.
- `show_description.py` — renders the benchmark file (merged with its sibling dataset file) as Markdown.
- `verify_description.py` — validates the benchmark jsonld file against `semantic_benchmark`.
- `workflow.py` — runs the scripts above as one CI-friendly pipeline.
- `utils.py` — shared string/number/file helpers.
- `metadata/` — `builder.py`, `graph.py`, `repo_source.py` (resolves `module_dir`: local path, or clone/update a git URL), `repository.py`, `parameters.py`, `metrics.py`, `publication.py`, `software.py` — benchmark.jsonld generation.
- `ai/` — `prompts.py`, `inference.py`, `validation.py`, `review.py` (interactive CLI review/edit step, run between inference and graph-build), `cache.py`, `corrections.py` — LLM-based semantic metadata inference and its cross-benchmark corrections memory.
- `snakefile/` — `generator.py`, `renderer.py` — executable-workflow generation.

## Common edge cases

- **No `--scenario-params` supplied**: falls back to an interactive prompt; unattended/CI runs must pass this flag explicitly or the run will hang waiting for input.
- **No `--skip-review`**: the review step also needs a real terminal (`input()`); unattended/CI runs must pass `--skip-review` or the run will hang there too, even with `--scenario-params` set.
- **Missing required Snakefile flags**: `--container-image` and `--container-shared-dir` are required even if you only care about the `benchmark.jsonld` output.
- **Verification failures**: a failing `verify_description.py` run should block downstream packaging/publishing — `workflow.py` already propagates this as a non-zero exit code.
- **Rotating-cylinder / radial-mesh benchmarks**: only enable `--mesh-split` (and its `--radial-cells-flag`/`--angular-cells-flag`/`--grading-flag`) when the benchmark actually uses a radial mesh; it's not a general-purpose flag.
- **Missing API key**: `GROQ_API_KEY` (or `OPENAI_API_KEY` with `--provider openai`) must be set before the inference step runs.
- **Still hitting rate limits despite `--inference-tpm-budget`**: set it well BELOW your account's actual TPM cap, not equal to it -- e.g. 8,000-9,000 on a 12,000 TPM account. Token estimation is a heuristic (chars/4 plus a per-item completion guess, see `ai.inference`'s `CHARS_PER_TOKEN_ESTIMATE`), and other activity sharing the same API key (a concurrent run, manual testing) eats into the same rolling window that budget is meant to protect.
- **A run uses a different model than expected**: `--model` always overrides `PROVIDER_CONFIG[provider]["default_model"]` -- check the actual command/script invocation (and any wrapper/CI job) for a lingering `--model` flag before assuming the default changed. The codebase has exactly one place that sets each provider's default (`ai.inference.PROVIDER_CONFIG`); nothing else in this repo hardcodes or overrides a model name.
- **`module_dir` as a git URL**: cloned into a throwaway temp directory by default, deleted once the run finishes -- nothing persists across runs, including the per-module inference cache, so each run against the same URL re-infers from scratch. Pass `--keep-clone` if you'd rather clone into a persistent cache directory and reuse it (fetch + checkout, not a full re-clone) -- and its `.parameter_metadata_cache.json`/`.metric_metadata_cache.json` -- on the next run against the same URL. Use `--fresh-clone` to discard and re-clone a persistent one if it ever gets into a bad state.

## License

MIT — see `LICENSES/MIT.txt`. Licensing follows the [REUSE](https://reuse.software) convention.
