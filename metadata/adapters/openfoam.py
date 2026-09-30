# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.adapters.openfoam

OpenFOAM: a case directory (system/controlDict, constant/, 0/) -- given
directly, or zipped next to the benchmark's Snakefile (as in
rotating-cylinders/openfoam/output_template.zip).

Inputs are dictionary entries: section = dictionary file relative to the
case ("constant/MRFProperties"), key = entry path ("MRF1/omega"). Template
placeholders such as "omega {omega};" mark the inputs a benchmark sets per
run; they're pre-selected and have no value of their own (the values then
come from a reference benchmark, see --reference-benchmark).

Metrics are the keys a post-processing script writes to
solution_metrics.json. The solver name comes from controlDict's
`application`.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from metadata.adapters.base import SoftwareAdapter
from metadata.foam_dict import FoamEntry, is_foam_dict, read_foam_dict
from metadata.metrics import METRICS_FILE_NAME, MetricCandidate, discover_metrics, discover_metrics_in_python
from metadata.parameters import ParameterCandidate
from utils import read_text, to_number

#: Where zipped template cases are unpacked, inside the module folder.
EXTRACT_DIR = ".benchmantic"
#: Case sub-folders whose dictionaries are inputs.
INPUT_DIRS = ("system", "constant", "0", "0.orig")
#: Never parameters: mesh/geometry data and generated folders.
SKIP_PARTS = {"polyMesh", "triSurface", "extendedFeatureEdgeMesh", "processor0", EXTRACT_DIR}
#: Entry paths that are bookkeeping or post-processing, not inputs.
SKIP_ENTRY = re.compile(r"^(functions/|dimensions$|libs$|convertToMeters$|scale$)")
#: Longest value (in tokens) still treated as a parameter (vectors, small lists).
MAX_VALUE_TOKENS = 9


def _is_case_dir(path: Path) -> bool:
    return (path / "system" / "controlDict").is_file()


def _zip_has_case(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return any(n.rstrip("/").endswith("system/controlDict") for n in zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def _case_dirs_under(root: Path) -> list[Path]:
    return sorted({p.parent.parent for p in root.rglob("controlDict")
                   if p.parent.name == "system" and EXTRACT_DIR not in p.parts})


def _case_zips_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.zip") if EXTRACT_DIR not in p.parts and _zip_has_case(p))


def _usage_lines(entry: FoamEntry, entries: list[FoamEntry], lines: list[str], limit: int = 8) -> list[str]:
    """Where an entry is referenced in its file ($/geom/nr, $nr), following
    one level of macros -- e.g. blockMeshDict's `blockInfo` uses $/geom/nr
    and each of the 4 hex blocks uses $blockInfo. Tells the LLM how the
    value is used (per block, per zone, ...)."""
    refs = {f"$/{entry.path}", f"${entry.path}", f"${entry.path.split('/')[-1]}"}

    def lines_referencing(names: set[str]) -> list[int]:
        pattern = re.compile("|".join(re.escape(n) + r"(?![\w/])" for n in names))
        return [i for i, text in enumerate(lines) if pattern.search(text) and i + 1 != entry.line]

    hits = lines_referencing(refs)
    macros = {f"${e.path}" for e in entries if "/" not in e.path and any(r in e.value for r in refs)}
    if macros:
        hits += [i for i in lines_referencing(macros) if i not in hits]
    return [lines[i].strip() for i in sorted(hits)[:limit]]


class OpenFOAMAdapter(SoftwareAdapter):
    name = "OpenFOAM"
    slug = "openfoam"
    input_label = "the OpenFOAM case dictionaries"
    default_selection_label = "template placeholders and physical-property entries"

    @classmethod
    def detect(cls, root: Path) -> bool:
        return _is_case_dir(root) or bool(_case_dirs_under(root)) or bool(_case_zips_under(root))

    # --- locating the case(s) --------------------------------------------
    def find_module_dir(self, root: Path, args: Any) -> Path:
        if _is_case_dir(root):
            return root
        cases, zips = _case_dirs_under(root), _case_zips_under(root)
        if len(cases) == 1 and not zips:
            return cases[0]
        homes = sorted({p.parent for p in cases} | {z.parent for z in zips})
        if len(homes) == 1:
            return homes[0]
        if not homes:
            sys.exit(f"Error: no OpenFOAM case (a folder with system/controlDict) found under {root}.")
        listing = "\n  ".join(str(h) for h in homes)
        sys.exit(f"Error: found OpenFOAM cases in several places under {root}:\n  {listing}\n"
                 "Point module_dir at the one you want.")

    def _extract_zip(self, zip_path: Path, module_dir: Path) -> Path:
        target = module_dir / EXTRACT_DIR / zip_path.stem
        stamp = target / ".extracted_from_mtime"
        mtime = str(zip_path.stat().st_mtime)
        if not (stamp.exists() and stamp.read_text() == mtime):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            stamp.write_text(mtime)
        # the zip may hold the case at its root or inside one folder
        return target if _is_case_dir(target) else next(iter(_case_dirs_under(target)), target)

    def discover_cases(self, module_dir: Path) -> list[tuple[Path, str]]:
        if _is_case_dir(module_dir):
            return [(module_dir, module_dir.name)]
        cases = [(d, d.name) for d in _case_dirs_under(module_dir)]
        cases += [(self._extract_zip(z, module_dir), z.stem) for z in _case_zips_under(module_dir)]
        if not cases:
            sys.exit(f"No OpenFOAM case found under {module_dir}")
        ids = [c[1] for c in cases]
        if len(set(ids)) != len(ids):
            sys.exit(f"Duplicate OpenFOAM case names under {module_dir}: {ids}")
        return cases

    def _template_case(self, module_dir: Path) -> Path:
        return self.discover_cases(module_dir)[0][0]

    # --- inputs -----------------------------------------------------------
    def _dictionary_files(self, case_dir: Path) -> list[Path]:
        files = []
        for sub in INPUT_DIRS:
            base = case_dir / sub
            if base.is_dir():
                files += [p for p in sorted(base.rglob("*"))
                          if p.is_file() and not (SKIP_PARTS & set(p.relative_to(case_dir).parts)) and is_foam_dict(p)]
        return files

    @staticmethod
    def _is_parameter(section: str, entry: FoamEntry) -> bool:
        if SKIP_ENTRY.search(entry.path):
            return False
        value = entry.value.strip()
        if entry.placeholder:
            return True
        if value.startswith(("$", "#")) or "$" in value:
            return False  # derived from another entry
        tokens = entry.tokens
        if tokens and tokens[0] == "uniform":
            tokens = tokens[1:]
        if not tokens or len(tokens) > MAX_VALUE_TOKENS:
            return False
        if all(re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$", t) for t in tokens):
            return True
        # single-word choices (models etc.) only in constant/
        return section.startswith("constant/") and len(tokens) == 1

    def discover_parameters(self, module_dir: Path) -> list[ParameterCandidate]:
        case_dir = self._template_case(module_dir)
        candidates = []
        for path in self._dictionary_files(case_dir):
            section = str(path.relative_to(case_dir))
            lines = read_text(path).splitlines()
            entries = read_foam_dict(path)
            for entry in entries:
                if not self._is_parameter(section, entry):
                    continue
                context = "\n".join(lines[max(0, entry.line - 3):entry.line + 2]).strip()
                usage = _usage_lines(entry, entries, lines)
                if usage:
                    context += "\n... used in:\n" + "\n".join(usage)
                if entry.placeholder:
                    hint = (f"template placeholder {{{entry.placeholder}}} in OpenFOAM dictionary {section}, "
                            f"entry {entry.path} -- set per benchmark run")
                else:
                    hint = f"OpenFOAM dictionary {section}, entry {entry.path}"
                candidates.append(ParameterCandidate(
                    section=section, key=entry.path, value=entry.value, tokens=entry.tokens,
                    cpp_hint=hint, code_context=context,
                ))
        return candidates

    def default_scenario_candidates(self, candidates):
        placeholders = [c for c in candidates if c.value.strip().startswith("{") and c.value.strip().endswith("}")]
        if placeholders:
            return placeholders
        chosen = [c for c in candidates
                  if (c.section.startswith("constant/") or c.section.startswith("system/blockMeshDict"))
                  and all(re.match(r"^[-+.\d eE]+$", t) for t in c.tokens)]
        return chosen or list(candidates)

    def resolve_case_params(self, case_dir: Path, parameter_fields: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, dict[str, FoamEntry]] = {}
        values: dict[str, Any] = {}
        placeholders = []
        for key, spec in parameter_fields.items():
            section, entry_path = spec["ini"]
            if section not in parsed:
                parsed[section] = {e.path: e for e in read_foam_dict(case_dir / section)}
            entry = parsed[section].get(entry_path)
            if entry is None:
                raise ValueError(f"In case '{case_dir}': {section} has no entry {entry_path}")
            if entry.placeholder:
                placeholders.append(f"{{{entry.placeholder}}} ({section}: {entry_path})")
                continue
            tokens = entry.tokens[1:] if entry.tokens[:1] == ["uniform"] else entry.tokens
            if spec.get("full_value") or len(tokens) > 1 and spec.get("index") is None:
                values[key] = [to_number(t) for t in tokens]
            else:
                index = spec.get("index", 0) or 0
                if not -len(tokens) <= index < len(tokens):
                    raise ValueError(f"In case '{case_dir}': {section} {entry_path} = '{entry.value}' "
                                     f"has no token {index}")
                values[key] = to_number(tokens[index])
        if placeholders:
            raise ValueError(
                f"In case '{case_dir}': these inputs are template placeholders with no value of their own: "
                + ", ".join(placeholders)
                + ". Pass --reference-benchmark <benchmark.jsonld> to take the configurations from an existing "
                  "benchmark description, or point module_dir at a case with concrete values."
            )
        return values

    # --- metrics ------------------------------------------------------------
    def _script_files(self, module_dir: Path, repo_root: Path) -> list[Path]:
        roots = [module_dir]
        if module_dir.parent != module_dir and repo_root in (module_dir.parent, *module_dir.parent.parents):
            roots.append(module_dir.parent)
        files: list[Path] = []
        for r in roots:
            pattern = r.rglob("*") if r == module_dir else r.glob("*")
            for p in pattern:
                if (p.is_file() and (p.suffix in (".py", ".sh") or p.name.startswith("Allrun"))
                        and not (SKIP_PARTS & set(p.parts)) and p not in files):
                    files.append(p)
        return files

    def _metrics_scripts(self, module_dir: Path, repo_root: Path) -> list[Path]:
        return [p for p in self._script_files(module_dir, repo_root) if METRICS_FILE_NAME in read_text(p)]

    def build_hints(self, module_dir: Path, repo_root: Path) -> dict[str, Any]:
        case_dir = self._template_case(module_dir)
        hints: dict[str, Any] = {"has_allrun": (case_dir / "Allrun").is_file(),
                                 "case_template_zip": next((z.name for z in _case_zips_under(module_dir)), None)}
        scripts = [p for p in self._metrics_scripts(module_dir, repo_root) if p.suffix == ".py"]
        if scripts:
            script = scripts[0]
            hints["metrics_script"] = script.name
            hints["metrics_args"] = "--single-case ." if "--single-case" in read_text(script) else "."
        return hints

    def discover_metrics(self, module_dir: Path, repo_root: Path) -> list[MetricCandidate]:
        found: list[MetricCandidate] = []
        seen: set[str] = set()
        for path in self._metrics_scripts(module_dir, repo_root):
            text = read_text(path)
            batch = discover_metrics_in_python(text) if path.suffix == ".py" else discover_metrics(text)
            for c in batch:
                if c.key not in seen:
                    seen.add(c.key)
                    found.append(c)
        if not found:
            sys.exit(f"No metrics found: expected a post-processing script next to the case that writes "
                     f"{METRICS_FILE_NAME} (e.g. json.dump({{'l2_error': ...}}, ...)).")
        return found

    # --- descriptive metadata -------------------------------------------------
    def _markdown_docs(self, module_dir: Path, repo_root: Path) -> list[Path]:
        docs = []
        for d in (module_dir, module_dir.parent, repo_root):
            docs += sorted(d.glob("README*.md")) + sorted((d / "docs").glob("*.md"))
        seen, out = set(), []
        for p in docs:
            if p.exists() and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
        return out

    def benchmark_description(self, module_dir: Path, texts: dict[str, str]) -> str:
        """Title + 'Problem description' section from the benchmark docs,
        e.g. docs/benchmark-documentation.md."""
        repo_root = getattr(self, "docs_root", None) or self._repo_root_hint or module_dir
        for doc in self._markdown_docs(module_dir, repo_root):
            text = read_text(doc)
            m = re.search(r"^#{1,3}\s*(problem description|description|overview)\s*\n(.*?)(?=^#{1,3}\s|\Z)",
                          text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if m:
                title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                body = re.sub(r"\s+", " ", m.group(2)).strip()
                return f"{title.group(1).strip()}\n\n{body}" if title else body
        return ""

    _repo_root_hint: Path | None = None

    def source_texts(self, module_dir: Path) -> dict[str, str]:
        case_dir = self._template_case(module_dir)
        return {str(p.relative_to(case_dir)): read_text(p)
                for p in self._dictionary_files(case_dir)
                if p.name in ("controlDict", "transportProperties", "physicalProperties", "MRFProperties")}

    def benchmark_label(self, module_dir: Path, repo_root: Path, texts: dict[str, str]) -> str | None:
        self._repo_root_hint = repo_root
        for doc in self._markdown_docs(module_dir, repo_root):
            m = re.search(r"^#\s+(.+)$", read_text(doc), re.MULTILINE)
            if m:
                title = re.sub(r"\s*\(.*?\)\s*", " ", m.group(1))          # drop "(Taylor-Couette Flow)"
                title = re.sub(r"\bbenchmark\b", "", title, flags=re.IGNORECASE)
                title = " ".join(title.split())
                if title:
                    return title.lower()
        return None

    def executable_name(self, module_dir: Path, repo_root: Path):
        self._repo_root_hint = repo_root
        case_dir = self._template_case(module_dir)
        control = case_dir / "system" / "controlDict"
        for e in read_foam_dict(control):
            if e.path == "application":
                return e.value.strip(), control
        return None, None
