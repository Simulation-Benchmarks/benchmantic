# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.autoconfig

Writes a software description with the LLM when benchmantic has none for
the software named in the run config (or only part of one). The LLM sees
the repository (file list, short excerpts, the contents of zip archives),
software/template.yaml and the descriptions in software_descriptions/ as
examples (one for the same software first, when there is one). A proposal
is only accepted after it has been tried on the repository -- the
implementation, its cases, inputs and metrics must actually be found;
otherwise the LLM gets the errors and tries again. The accepted description
is cached next to the benchmark (.benchmantic_software.yaml) and reused
while the run config stays the same; with the user's permission it is also
saved to software_descriptions/, so the next run naming that software uses
it directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

CACHE_NAME = ".benchmantic_software.yaml"
ATTEMPTS = 3

SKIP_DIRS = {".git", "__pycache__", ".benchmantic", "build", "build-cmake", "node_modules", ".venv", "outputs"}
EXCERPT_SUFFIXES = {".input", ".ini", ".cfg", ".conf", ".yaml", ".yml", ".json", ".toml", ".py", ".sh",
                    ".cc", ".cpp", ".hh", ".h", ".hpp", ".md", ".txt", ".m", ".jl", ".f90", ".prm"}
EXCERPT_NAMES = {"CMakeLists.txt", "Allrun", "Makefile", "controlDict", "Dockerfile"}
MAX_LISTED = 150
MAX_EXCERPT_FILES = 20
EXCERPT_LINES = 40
EXCERPT_BUDGET = 9000

SYSTEM_PROMPT = """\
You configure "benchmantic", a tool that writes a semantic description and
a Snakemake workflow for a simulation benchmark implemented in some
simulation software. benchmantic needs a small "software description":
which files mark the implementation, which file marks one case
(configuration), the input file format, where the metrics are written,
where the benchmark's name and description come from, and how one case is
run in a container.

You are given: the documented template (every allowed key, with comments),
complete working descriptions of other software as examples, the settings
the user has already fixed (possibly none -- then also choose `name`), and
a view of the repository. Return the complete description for the user's
software as JSON.

Rules:
- Use only keys documented in the template; values must use only the
  options listed there (formats: ini, foam_dict; run.type: program,
  case_folder; label: markdown_title, cpp_class; description:
  markdown_section, doxygen; executable.from: cmake, entry, none).
- Keep every setting the user fixed exactly as given.
- All paths are relative to the repository folder shown, and every file you
  name must exist there (or inside a zip listed there, for case files).
- Choose files and patterns that match THIS repository; the examples only
  show the format.
- `notes`: two to six sentences on what a parameter's section/key is in
  this software and which names imply which units.

Respond with a raw JSON array holding exactly ONE object (the description), no
markdown fences and no prose.
"""

PROMPT_TEMPLATE = """\
=====================
template (every allowed key, documented)
=====================
{template}

{examples}

=====================
settings the user fixed (keep them)
=====================
{user_json}

=====================
repository: {root_name} -- files
=====================
{listing}

=====================
repository -- excerpts
=====================
{excerpts}
{errors}
Return the JSON array with the one description object now.
"""


#: At most this many example descriptions go into the prompt.
MAX_EXAMPLES = 3


def _examples(name: str | None) -> str:
    """The descriptions in software_descriptions/, one for the same software first."""
    from software import DESCRIPTIONS_DIR

    files = sorted(DESCRIPTIONS_DIR.glob("*.yaml"))
    wanted = (name or "").strip().lower()

    def same_software(p: Path) -> bool:
        head = p.read_text(encoding="utf-8")
        m = re.search(r"^name:\s*(.+)$", head, re.MULTILINE)
        return bool(wanted) and (p.stem.lower() == wanted or (m and m.group(1).strip().lower() == wanted))

    files = sorted(files, key=lambda p: not same_software(p))[:MAX_EXAMPLES]
    return "\n".join(f"=====================\nexample {i}: {p.stem}\n=====================\n"
                     f"{p.read_text(encoding='utf-8')}" for i, p in enumerate(files, 1))


def user_hash(user: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(user, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _listing(root: Path) -> tuple[list[str], list[Path]]:
    lines: list[str] = []
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if set(rel.parts) & SKIP_DIRS or any(part.startswith(".") and part != ".benchmantic" for part in rel.parts):
            continue
        if p.is_file():
            files.append(p)
            if len(lines) < MAX_LISTED:
                lines.append(str(rel))
                if p.suffix == ".zip":
                    try:
                        with zipfile.ZipFile(p) as zf:
                            names = [n for n in zf.namelist() if not n.endswith("/")]
                    except (zipfile.BadZipFile, OSError):
                        names = []
                    lines += [f"    [inside {rel.name}] {n}" for n in names[:40]]
    if len(files) > MAX_LISTED:
        lines.append(f"... and {len(files) - MAX_LISTED} more files")
    return lines, files


def _excerpts(root: Path, files: list[Path]) -> str:
    def rank(p: Path) -> tuple:
        name = p.name.lower()
        return (not name.startswith("readme"), p.name not in EXCERPT_NAMES, len(p.relative_to(root).parts), str(p))

    chosen = [p for p in files if p.suffix in EXCERPT_SUFFIXES or p.name in EXCERPT_NAMES]
    out, used = [], 0
    for p in sorted(chosen, key=rank)[:MAX_EXCERPT_FILES]:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        text = "\n".join(lines[:EXCERPT_LINES])
        if used + len(text) > EXCERPT_BUDGET:
            break
        used += len(text)
        more = f"\n    ... ({len(lines) - EXCERPT_LINES} more lines)" if len(lines) > EXCERPT_LINES else ""
        out.append(f"--- {p.relative_to(root)}\n{text}{more}")
    return "\n\n".join(out)


def check(spec: dict[str, Any], root: Path, args: Any) -> list[str]:
    """Try `spec` on the repository; [] when everything is found."""
    from software.formats import FORMATS
    from software.generic import RUN_TYPES, GenericAdapter

    problems: list[str] = []
    if not spec.get("name"):
        problems.append("`name` is missing")
    if (spec.get("inputs") or {}).get("format") not in FORMATS:
        problems.append(f"inputs.format must be one of {', '.join(FORMATS)}")
    if not (spec.get("cases") or {}).get("marker"):
        problems.append("cases.marker is missing")
    run = spec.get("run") or {}
    run_type = run.get("type")
    if run_type not in RUN_TYPES:
        problems.append(f"run.type must be one of {', '.join(RUN_TYPES)}")
    required = {"program": ("command", "input_argument"),
                "case_folder": ("command", "command_without_script", "set_value_command")}.get(run_type, ())
    missing = [k for k in required if not run.get(k)]
    if missing:
        problems.append(f"run ({run_type}) needs: {', '.join(missing)}")
    if problems:
        return problems

    try:
        cls = GenericAdapter.from_spec(spec, "candidate")
        if not cls.detect(root):
            return [("the implementation is not recognised in the repository with these settings "
                     "(implementation.files / cases.marker do not match any file)")]
        adapter = cls()
        module_dir = adapter.find_module_dir(root, args)
        cases = adapter.discover_cases(module_dir)
        if not cases:
            problems.append("no cases found")
        if not adapter.discover_parameters(module_dir):
            problems.append(f"no input parameters found in the cases' {spec['inputs'].get('files') or 'input files'}")
        if not adapter.discover_metrics(module_dir, root):
            problems.append("no metrics found")
    except SystemExit as exc:
        problems.append(str(exc.code))
    except Exception as exc:  # noqa: BLE001 -- any failure is feedback for the next attempt
        problems.append(f"{type(exc).__name__}: {exc}")
    return problems


def _read_cache(path: Path, wanted_hash: str) -> dict[str, Any] | None:
    import yaml

    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^# user-settings: (\w+)$", text, re.MULTILINE)
    if not m or m.group(1) != wanted_hash:
        return None
    spec = yaml.safe_load(text)
    return spec if isinstance(spec, dict) else None


def _write_cache(path: Path, spec: dict[str, Any], wanted_hash: str, root: Path) -> None:
    import yaml

    header = (f"# Software description for {spec.get('name')}, written by the LLM for {root.name} and checked\n"
              "# against it by benchmantic. Review it; to reuse it elsewhere, point a run\n"
              "# config's `software-description:` at a copy of it. Delete this file to get a\n"
              "# new proposal.\n"
              f"# user-settings: {wanted_hash}\n\n")
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100),
                    encoding="utf-8")


def complete(root: Path, settings: dict[str, Any], args: Any) -> dict[str, Any]:
    """The full software description: `settings` (what is already known,
    kept as given) plus the LLM's proposal for everything else, checked on
    the repository."""
    from ai.inference import DEFAULT_PROVIDER, DEFAULT_TPM_BUDGET, _query_llm_json_array
    from software.generic import merge_specs

    label = settings.get("name") or "this benchmark"
    wanted = user_hash(settings)
    cache = root / CACHE_NAME
    if not getattr(args, "clear_cache", False):
        cached = _read_cache(cache, wanted)
        if cached is not None:
            print(f"Using the software description for {cached.get('name')} from {cache} (0 API queries).")
            return cached

    listing, files = _listing(root)
    base_prompt = PROMPT_TEMPLATE.format(
        template=(Path(__file__).resolve().parent / "template.yaml").read_text(encoding="utf-8"),
        examples=_examples(settings.get("name")),
        user_json=json.dumps(settings, indent=2),
        root_name=root.name,
        listing="\n".join(listing),
        excerpts=_excerpts(root, files),
        errors="{errors}",
    )

    def one_object(data: Any) -> list[dict]:
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError("Expected a JSON array holding exactly one description object.")
        return data

    provider = getattr(args, "provider", None) or DEFAULT_PROVIDER
    errors_text = ""
    problems: list[str] = []
    spec: dict[str, Any] = {}
    for attempt in range(1, ATTEMPTS + 1):
        print(f"\nAsking {provider} to write the software description for {label} "
              f"(attempt {attempt}/{ATTEMPTS})...")
        proposal = _query_llm_json_array(
            provider=provider,
            model=getattr(args, "model", None),
            system_prompt=SYSTEM_PROMPT,
            prompt=base_prompt.replace("{errors}", errors_text),
            item_count=1,
            item_label="software description",
            validator=one_object,
            retries=3,
            retry_delay=2.0,
            verbose=getattr(args, "verbose", False),
            debug=getattr(args, "debug", False),
            tpm_budget=getattr(args, "inference_tpm_budget", DEFAULT_TPM_BUDGET),
            allow_model_fallback=not getattr(args, "no_model_fallback", False),
        )[0]
        spec = merge_specs(proposal, settings)
        problems = check(spec, root, args)
        if not problems:
            _write_cache(cache, spec, wanted, root)
            print(f"Software description for {spec['name']} accepted after {attempt} attempt(s) "
                  f"and saved to {cache} -- review it there.")
            offer_to_save(spec, getattr(args, "save_software_description", None))
            return spec
        print("  the proposal did not work on the repository:\n    - " + "\n    - ".join(problems))
        errors_text = ("\n=====================\nyour previous proposal and why it failed -- fix it\n"
                       "=====================\n" + json.dumps(proposal, indent=2) + "\n\nproblems:\n- "
                       + "\n- ".join(problems) + "\n")

    failed = cache.with_name(CACHE_NAME.replace(".yaml", ".failed.yaml"))
    _write_cache(failed, spec, wanted, root)
    sys.exit(f"Error: no working software description for {label} after {ATTEMPTS} attempts. "
             f"The last proposal is in {failed}; fix it and point the run config's "
             "`software_description:` at it.")


def offer_to_save(spec: dict[str, Any], save: bool | None) -> None:
    """Save an accepted description to software_descriptions/<name>.yaml --
    only with the user's permission: `save-software-description: true` in
    the run config, or a yes when asked on a terminal. Never overwrites an
    existing description."""
    import yaml

    from software import DESCRIPTIONS_DIR

    slug = re.sub(r"[^0-9a-z]+", "-", str(spec["name"]).lower()).strip("-") or "software"
    target = DESCRIPTIONS_DIR / f"{slug}.yaml"
    if target.exists() or save is False:
        return
    if save is None:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return  # nobody to ask (or the question wouldn't be shown)
        answer = input(f"Save this description as {target}, so `software: {spec['name']}` uses it "
                       "next time (and the LLM sees it as an example)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return
    header = ("# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks\n#\n# SPDX-License-Identifier: MIT\n#\n"
              f"# {spec['name']}: written by the LLM and checked on a benchmark.\n\n")
    target.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100),
                      encoding="utf-8")
    print(f"Saved: {target}")
