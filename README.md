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
- Interactive review step: before anything is built into the graph, parameters AND metrics go through one combined review pass. On a real terminal, this is a curses screen showing only the items that actually need a look (low confidence, or a structural flag unrelated to the AI's own confidence) — everything else is auto-accepted and just counted, with Accept/Rename/Change unit/Change type/Change quantity kind/Edit explanation per item, chosen with arrow keys or a single digit press (`1`-`7`); falls back automatically to a plain-text table (same idea, `--skip-review`-friendly) when curses isn't usable (no real TTY, e.g. piped/CI output) or you back out of it with `q`.
- Quiet by default: normal output is a handful of `✓`/`○` checklist lines plus a final boxed summary (benchmark name, parameter/metric/case counts, validation status, generated files, the next command to run) instead of a raw dump of every discovery/inference detail. Pass `-v`/`--verbose` for the detail behind that (resolved paths, per-request LLM headers), or `--debug` for the exact request/response text of every LLM call on top of that.
- The interactive run is framed as six numbered steps (`1/6 Discover benchmark` … `6/6 Validate`), each announced with a short banner, so it's always clear which part of the pipeline is running -- these banners print regardless of `--verbose`/`--debug`. First run against a real terminal opens with a one-screen explanation of what the tool produces ("Press Enter to continue…"); non-interactive/CI runs (`--scenario-params` + `--skip-review`, or non-TTY) skip it automatically.
- Outputs step: which artifacts to actually write is its own explicit decision, not an afterthought. The benchmark description AND the review report are always generated, no matter what; only the dataset-provenance sidecar and the Snakefile are real choices. On a real terminal you're shown each optional artifact with a one-line purpose ("Reproducible workflow -- Snakemake workflow for executing the benchmark") and its filename, offered presets (Standard / Snakefile / With dataset / Description only / Custom) each with its own one-line description of what it actually generates (e.g. Snakefile: "Generates the semantic description and review report for the corresponding benchmark, plus a reproducible Snakemake workflow -- no separate dataset-provenance file."), then a preview of exactly what's about to be written with a final `? Generate these files? [Y/n]` confirmation before anything is generated. The choice is remembered per module (`.benchmantic_outputs.json`, beside the existing `.parameter_metadata_cache.json`) so a later run against the same module doesn't ask again; override any run with `--outputs <preset>` or `--outputs dataset,snakefile` (see `--help` for the full preset list), which also skips the prompt and preview entirely -- useful for CI.
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

`--config` (see the Config files section below) additionally needs `pyyaml`, only if you actually use that flag:

```bash
pip install pyyaml
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

The review step covers parameters and metrics together, in one pass. On a real terminal it's a curses screen: a summary ("N accepted automatically, M require your review") with explicit choices -- `[F]` review just the flagged items (or just press Enter), `[A]` also review the N auto-accepted ones (the reviewer chooses this, it's not automatic; shown even on a run where nothing happened to get flagged, so there's always a chance to double-check), `[N]` accept everything with no review, `[Q]` quit to the plain-text fallback. Whichever set you review, each item gets an Accept/Rename/Change unit/Change type/Change quantity kind/Edit explanation menu — arrow keys (or `j`/`k`), or the action's own number (`1`-`7`, shown next to it), move the highlight to that action right away so you can see where you've landed, and Enter confirms whatever's currently highlighted (a stray keypress just moves the highlight, it never fires an action on its own). When you pick "Edit explanation" (or any other text field), the prompt shows the field's current value inline and can run long -- it's word-wrapped across as many lines as the terminal's actual width needs, with the line you type into always placed on its own row below the wrapped prompt, so nothing is clipped and the cursor is never pinned against the edge of a narrow window. The Outputs step's preset descriptions (below) wrap the same way. Where curses isn't usable (piped output, no TTY, `q` to back out), it falls back to a plain-text table per kind: type a row number to edit a field, or press Enter to accept the table as shown — rows below `--review-confidence-threshold` (default `0.7`) or flagged for a structural reason (e.g. a repaired mapping) show a `!`/`?` marker either way; this fallback doesn't have a separate "review the accepted ones" prompt since every row in its table is already visible and editable at once.

Normal output stays short — a handful of `✓ ...`/`○ ...` checklist lines through discovery and inference, then a boxed "Benchmark generated successfully" summary at the end. Pass `-v`/`--verbose` to see what's collapsed into those checklist lines (resolved paths, per-LLM-call request headers, RO-Crate's full failing-checks list), or `--debug` for the exact request/response text of every LLM call as well. The run itself is announced step by step (`1/6 Discover benchmark`, `2/6 Select parameters`, ...) regardless of verbosity, and — the first time, on a real terminal — opens with a short "Press Enter to continue…" explanation of what the tool produces.

Right before generation, an Outputs step asks (or, after the first run against a module, silently reuses) which of the dataset sidecar / Snakefile to actually produce -- the benchmark description and `review.md` are always generated, not a choice -- with a preview and `[Y/n]` confirmation; pass `--outputs standard|snakefile|dataset|description-only|none|dataset,snakefile` to set it non-interactively (see `--help`) instead of being asked.

For non-interactive/CI runs, add `--skip-review` (and make sure `--scenario-params` is always passed, since omitting it also triggers an interactive prompt) — this also skips the intro banner and the Outputs prompt/preview, silently reusing the module's last saved output selection (or generating everything, the first time). Skip individual pipeline steps with `--skip-show` / `--skip-check`. Run any script with `--help` for its full option list.

### Config files

Instead of retyping a benchmark's usual flag set every run, save it to a YAML file and pass `--config <path>` (accepted by both `workflow.py` and `describe_benchmark.py`, and any flag also given explicitly on the command line still overrides the same key in the file):

```yaml
# rotating-cylinders.yaml
module_dir: /Users/sarbani/NFDI_Benchmark/dumux_test/rotating-cylinders
full-value-params: Radial0
container-image: git.iws.uni-stuttgart.de:4567/benchmarks/rotating-cylinders:3.1
container-shared-dir: /dumux/shared
mesh-split: true
zip-name-flag: Problem.Name
semantic-benchmark-src: ../semantic-benchmark
verbose: true
review-confidence-threshold: 0.99
clear-cache: true
```

```bash
python3 workflow.py --config rotating-cylinders.yaml
```

Any flag either script accepts can go in the file (keys use the flag name with or without the leading `--`, dashes or underscores both fine) — see `config.py`'s module docstring for the full format and two argparse-inherent caveats (a `true` boolean can't be forced back to `false` from the command line for one run; an `append`-style flag like `--exclude-flag` given in both places gets both, not one replacing the other). An unrecognized key in the file is a clear startup error, not a silent no-op; `module_dir`/`--container-image`/`--container-shared-dir` are the only flags actually required to end up set (from either source) before the run starts.

## Architecture / Pipeline walkthrough

`describe_benchmark.py` is the single entry point, and runs five stages in process (calling `metadata.builder` and `snakefile.generator` directly, not via subprocess) — on screen, these map onto six numbered step banners (`1/6 Discover benchmark`, `2/6 Select parameters`, `3/6 Review semantic annotations`, `4/6 Select outputs`, `5/6 Generate`, `6/6 Validate`), since stage 1 below covers both discovery and parameter selection:

1. **Discover** (`metadata.repo_source`, `metadata.repository`, `metadata.parameters`, `metadata.metrics`) — first resolves `module_dir`: a local path is used as-is; a git URL is cloned into a throwaway temp directory, deleted automatically once the run finishes (or, with `--keep-clone`, into a persistent cache directory instead — default `.benchmantic_repo_cache/`, override with `BENCHMANTIC_REPO_CACHE` or `--clone-dir`). Then scans the resolved module for parameter and metric candidates, and scrapes license, dependency, build, citation, and author information from source and README. Parameters can be selected via `--scenario-params`, or interactively if omitted -- on a real terminal, a curses checklist (`↑`/`↓`/`j`/`k` to move, `Space` to toggle the highlighted row, or type a row's own number to jump the highlight straight to it -- works for any row, not just the first few visible on screen, and the list scrolls automatically to fit however tall the terminal is; each digit you type appears on its own dedicated "Go to #: " line with a real, visible cursor, and the highlight moves live as you type, well before you press anything to confirm; a one-line `↑/↓: move, space: toggle, type a number: to jump` legend sits right under the title, visible immediately instead of scrolled past a long list to find, while state that changes as you go -- the running selected count, that "Go to #:" line, and the less-frequent `a`/`n`/`q` shortcuts -- prints right after the list instead, closer to where you're already looking once you've scanned through it; every line is independently word-wrapped to the terminal's actual width, so an option's own text (e.g. "continue") is never the part that gets cut off on a narrower window; press Enter to toggle the highlighted row, `a`/`n` to select all/none, Enter with nothing typed to finish, `q`/Escape to cancel to the plain-text prompt below); the plain-text fallback takes comma-separated index numbers, `all`, `none`, or `table` to reprint the list, with its column count adapted to the terminal's actual width so long parameter names don't wrap mid-row.
2. **Infer** (`ai.cache`, `ai.inference`) — for every candidate not already cached, asks an LLM to infer its physical meaning. This is broken down further inside the `ai/` package:
   - `ai.prompts` builds the prompt from the benchmark's description, the candidate parameters/metrics, and any relevant prior human corrections (see `ai.corrections` below) — deliberately *not* the full `main.cc`/`problem.hh` source. Each parameter already carries its own `getParam<>()` call-site excerpt (`metadata.parameters.attach_cpp_hints`) and each metric its own surrounding-code snippet (`metadata.metrics.discover_metrics`), which is far smaller and just as code-grounded; the full file is only searched as a last-resort fallback for the rare item with no per-item context of its own.
   - `ai.inference` queries the configured provider (Groq by default, or OpenAI), retrying up to 3 times on failure, and failing fast (no pointless retries/waits) on an unrecoverable size/rate-limit response instead of exhausting all 3 attempts on something retrying can't fix. Before any of that: more than `--inference-batch-size` (default 8) not-yet-cached items are split into multiple independent requests, each chunk is split *again* if its own estimated token cost (prompt + expected completion) exceeds `--inference-tpm-budget` (default 9,000) regardless of item count, and every request -- across the whole run, not just within one batch -- reserves its estimated cost against one shared tokens-per-minute ledger before sending, sleeping if needed so the run's real request rate stays under that budget too.
   - `ai.validation` checks the response for missing fields, out-of-range indices, and unit/quantityKind mix-ups, correcting or flagging them. If a response has a unit but no quantityKind, it's backfilled from the unit alone via two best-effort live lookups, in order: first the unit's own [QUDT vocabulary entry](http://qudt.org/vocab/unit/) (e.g. `unit:PA` -> `http://qudt.org/vocab/unit/PA`), asking QUDT directly what quantity kind that unit belongs to; then, only if that doesn't resolve it, the [SI Digital Framework](https://si-digital-framework.org/SI/units?lang=en) units page. Either way the result is flagged for review, since it's this pipeline's own inference rather than the model's.
   - `ai.cache` persists the final (post-review) result beside the module, so each item is inferred only once.
3. **Review** (`ai.review`, `ai.tui`) — parameters and metrics reviewed together in one combined pass (`review_or_skip_combined()`), not two separate sessions. On a real terminal this is `ai.tui`'s curses queue: a summary of how many items were auto-accepted vs. flagged (low confidence, or a structural issue like a positionally-reconstructed `ini`/`key` field -- see `ai.validation`), then just the flagged items, one per screen, with an Accept/Rename/Change unit/Change type/Change quantity kind/Edit explanation action menu. Wherever curses isn't usable (no real TTY, not importable, or the reviewer backs out with `q`) it falls back to `ai.review`'s plain-text table per kind -- one prompt, type a row number to edit any field or press Enter to accept the table as shown, `!`/`?` markers on rows worth a look but Enter always accepts regardless. Skipped entirely with `--skip-review`. Every value actually changed here (either UI) is recorded by `ai.corrections`, which persists it (matched by exact or fuzzy key similarity) so a similarly named/typed parameter in a *different* benchmark benefits from it too, next time it's inferred.
4. **Build** (`metadata.builder`) — wraps each reviewed value, along with the scraped manifest info, into typed nodes inside one metadata4ing `@graph` (`GraphBuilder`). Author, publisher, and software dependencies are assembled into a *separate* graph here too (`GraphBuilder.build_dataset_graph()`) rather than merged into the same one -- see the next step. Immediately before this, the **Select outputs** step (`metadata.builder._resolve_outputs_selection()`, `ai.tui.output_picker()`) decides which of the dataset sidecar and Snakefile to actually produce this run -- `--outputs`, if given, wins outright; otherwise an interactive picker (curses, or a plain-text preset menu) offers presets or a per-artifact custom choice, previews exactly what's about to be written, and asks `? Generate these files? [Y/n]` before anything is generated (answering `n` reopens the picker); otherwise (CI / `--skip-review` / no real terminal) the module's last saved choice is silently reused. Either way the resolved selection is persisted to `.benchmantic_outputs.json` beside the module (same convention as the parameter/metric inference caches) so a later run doesn't ask again. The benchmark description and the review report are never optional -- only the dataset sidecar and Snakefile are.
5. **Validate & Generate** (`metadata.builder`, `snakefile.generator`, `snakefile.renderer`) — checks the benchmark graph against the RO-Crate 1.1 profile, then writes `<benchmark-name>_benchmark.jsonld` (parameters, metrics, processing steps, RO-Crate root) always, plus whichever of a sidecar `<benchmark-name>_dataset.jsonld` (author, publisher, software dependencies, linked back to the benchmark file via `schema:isPartOf`) and a plain `Snakefile` (not zipped) the Outputs step above selected.

Two additional scripts sit downstream of generation:

- **`show_description.py`** loads the benchmark file, auto-discovers its sibling `_dataset.jsonld` file next to it (or takes `--dataset-jsonld` explicitly), and renders both together as Markdown tables (manifest info, dependencies, parameters, metrics) for human review, flagging any unit/quantityKind mix-ups. Unlike the interactive review step, this reflects the fully built graph — including manifest fields (license, authors, dependencies) and resolved case values that review never touches — and is saved to disk (`review.md`) as a durable record. `describe_benchmark.py` itself always calls this in-process (same as `workflow.py` does downstream), the same way it always writes the benchmark description itself -- it's not one of the Outputs step's choices. If the dataset sidecar wasn't generated that run, `review.md` still gets written, just without the manifest/dependency sections that come from it -- covering only the benchmark's own input parameters and output metrics.
- **`verify_description.py`** loads the benchmark file (not the dataset sidecar -- it has no bearing on the check) with the real `semantic_benchmark.BenchmarkLoader` and checks that every processing step, parameter set, and metric field mapping actually resolves — not just that the file is valid JSON-LD.

To fix a value after the fact (e.g. a run done with `--skip-review`, or something spotted later in `review.md`): edit the relevant entry in `.parameter_metadata_cache.json` (or `.metric_metadata_cache.json`) inside the module directory, then re-run `describe_benchmark.py` **without** `--clear-cache` so the fix is reused rather than re-inferred, and re-run `verify_description.py` to confirm.

## Repository structure

```
benchmantic/
├── describe_benchmark.py    # entry point: orchestrates metadata + Snakefile generation
├── show_description.py       # renders the benchmark + dataset jsonld files as Markdown for review
├── verify_description.py     # validates the benchmark jsonld file against semantic_benchmark
├── workflow.py                # runs the three scripts above as one CI-friendly pipeline
├── config.py                   # optional --config YAML support (workflow.py + describe_benchmark.py)
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
│   ├── review.py                #   combined review/edit step (plain-text) with confidence gating
│   ├── tui.py                    #   optional curses UI for parameter selection + the review queue
│   ├── cache.py                 #   on-disk inference caching + Outputs-step selection persistence (per module)
│   └── corrections.py           #   cross-benchmark human-correction memory
└── snakefile/                  # executable-workflow generation
    ├── generator.py             #   derives Snakefile inputs from benchmark.jsonld
    └── renderer.py               #   renders the actual Snakefile text
```

## License

MIT — see [`LICENSES/MIT.txt`](LICENSES/MIT.txt). Licensing follows the [REUSE](https://reuse.software) convention; run `reuse lint` to verify compliance.
