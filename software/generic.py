# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.generic

The one adapter behind every software config. A config only says *what*
an implementation looks like (which files mark it, where the cases are, the
input format, where metrics are written, how a case is run -- see
software/template.yaml); the steps themselves are the same for every code
and live here, in software/sources.py, software/formats/ and
snakefile/styles/. The config comes from the run config (software.resolve).
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

from metadata.metrics import METRICS_FILE_NAME, MetricCandidate, discover_metrics, discover_metrics_in_python
from metadata.parameters import ParameterCandidate
from snakefile.styles import get_style
from software import param_files, sources
from software.base import SnakefileContext, SnakefileResult, SoftwareAdapter
from software.formats import get_format
from utils import read_text

#: Where zipped template cases are unpacked, inside the implementation folder.
EXTRACT_DIR = ".benchmantic"
#: With `cases.files`, the name of each case's file in its own folder (plus the suffix).
CASE_FILE_STEM = "case_parameters"
#: `run.type` in a description -> the Snakefile style (snakefile/styles/).
RUN_TYPES = {"program": "command_line", "case_folder": "case_files", "script": "python_script",
             "parameter_file": "parameter_file"}


def load_spec(path: Path, require_name: bool = True) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        sys.exit(f"Error: reading {path} needs PyYAML (pip install pyyaml).")
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(spec, dict) or (require_name and not spec.get("name")):
        sys.exit(f"Error: {path} must be a YAML mapping with at least a `name`.")
    return spec


def merge_specs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` on top of `base`, section by section."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_specs(merged[key], value)
        else:
            merged[key] = value
    return merged


class GenericAdapter(SoftwareAdapter):
    spec: dict[str, Any] = {}
    #: Repository root last seen (label/executable); a fallback for the docs.
    _repo_root: Path | None = None
    #: The file a link pointed at (e.g. .../blob/main/chapter2/ns_code2.ipynb),
    #: set by find_module_dir(); None when the link named a folder.
    main_file: Path | None = None

    @classmethod
    def from_spec(cls, spec: dict[str, Any], slug: str | None = None) -> type[GenericAdapter]:
        """An adapter class for a software config."""
        slug = slug or re.sub(r"[^0-9a-z]+", "-", str(spec["name"]).lower()).strip("-") or "software"
        selection = (spec.get("inputs") or {}).get("default_selection") or {}
        attrs = {
            "spec": spec,
            "name": spec["name"],
            "slug": slug,
            "input_label": spec.get("input_label", f"the {spec['name']} input files"),
            "default_selection_label": selection.get("label", "all parameters"),
            "default_container_shared_dir": (spec.get("run") or {}).get("results_folder"),
        }
        class_name = re.sub(r"\W", "", spec["name"]) + "Adapter"
        return type(class_name, (cls,), attrs)

    # --- settings ---------------------------------------------------------------
    @classmethod
    def _section(cls, key: str) -> dict[str, Any]:
        return cls.spec.get(key) or {}

    @property
    def run_settings(self) -> dict[str, Any]:
        """The description's `run:` section (how one case is run)."""
        return self._section("run")

    @classmethod
    def style(cls) -> ModuleType:
        return get_style(RUN_TYPES.get(cls._section("run").get("type", "program"), "unknown"))

    @property
    def _format(self) -> ModuleType:
        return get_format(self._section("inputs").get("format", "ini"))

    # --- cases ---------------------------------------------------------------------
    def _marker(self) -> tuple[str, ...]:
        """The file marking a case: the configured one, or the parameter
        file found by search when the configured one isn't there."""
        if self._case_files_pattern():
            return (self._case_file_name(),)        # each case file, copied to a folder of its own
        marker = self._section("cases").get("marker")
        if not marker and self.main_file is not None:
            return (self.main_file.name,)          # the linked file marks the one case
        return getattr(self, "_found_marker", None) or Path(marker).parts

    # --- one case per file (`cases.files`) -----------------------------------------------
    def _case_files_pattern(self) -> str | None:
        return self._section("cases").get("files")

    def _case_file_name(self) -> str:
        """The name each case file gets in its own folder."""
        return CASE_FILE_STEM + Path(self._case_files_pattern() or "").suffix

    def case_files_folder(self, module_dir: Path) -> Path:
        """Where the case files are: `cases.folder`, relative to the implementation."""
        return (module_dir / self._section("cases").get("folder", ".")).resolve()

    def case_file_paths(self, module_dir: Path) -> list[Path]:
        pattern = self._case_files_pattern()
        folder = self.case_files_folder(module_dir)
        return sorted(p for p in folder.glob(pattern) if p.is_file()) if pattern and folder.is_dir() else []

    def _case_id_from_file(self, path: Path) -> str:
        """parameters_05.json with pattern parameters_*.json -> "05"."""
        pattern = self._case_files_pattern() or ""
        if pattern.count("*") == 1:
            head, tail = pattern.split("*")
            name = path.name
            if name.startswith(head) and name.endswith(tail) and len(name) > len(head) + len(tail):
                return name[len(head):len(name) - len(tail)]
        return path.stem

    def _file_cases(self, module_dir: Path) -> list[tuple[Path, str]]:
        """Each file matching `cases.files` is a case: it is copied to
        .benchmantic/cases/<id>/ under the implementation, where the input
        format reads it like any case folder. `case_sources` keeps the originals."""
        self.case_sources: dict[str, Path] = {}
        cases = []
        for path in self.case_file_paths(module_dir):
            case_id = self._case_id_from_file(path)
            target = module_dir / EXTRACT_DIR / "cases" / case_id
            target.mkdir(parents=True, exist_ok=True)
            copy = target / self._case_file_name()
            if not copy.exists() or copy.read_bytes() != path.read_bytes():
                copy.write_bytes(path.read_bytes())
            self.case_sources[case_id] = path
            cases.append((target, case_id))
        if not cases:
            sys.exit(f"Error: no {self.name} case file ({self._case_files_pattern()}) found in "
                     f"{self.case_files_folder(module_dir)}")
        return cases

    def _is_case(self, folder: Path) -> bool:
        return folder.joinpath(*self._marker()).is_file()

    def _case_dirs_under(self, root: Path) -> list[Path]:
        """Folders holding the case marker, in the order of the marker paths."""
        parts = self._marker()
        found = []
        for p in sorted(root.rglob(parts[-1])):
            if p.parts[-len(parts):] == parts and EXTRACT_DIR not in p.parts:
                case_dir = p.parents[len(parts) - 1]
                if case_dir not in found:
                    found.append(case_dir)
        return found

    def _zip_has_case(self, path: Path) -> bool:
        marker = "/".join(self._marker())
        try:
            with zipfile.ZipFile(path) as zf:
                return any(n.rstrip("/").endswith(marker) for n in zf.namelist())
        except (zipfile.BadZipFile, OSError):
            return False

    def _zips_under(self, root: Path) -> list[Path]:
        if not self._section("cases").get("zips"):
            return []
        return sorted(p for p in root.rglob("*.zip") if EXTRACT_DIR not in p.parts and self._zip_has_case(p))

    def case_zips(self, module_dir: Path) -> list[Path]:
        return self._zips_under(module_dir)

    def _extract_zip(self, zip_path: Path, module_dir: Path) -> Path:
        target = module_dir / EXTRACT_DIR / zip_path.stem
        stamp = target / ".extracted_from_mtime"
        mtime = str(zip_path.stat().st_mtime)
        if not (stamp.exists() and stamp.read_text() == mtime):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            stamp.write_text(mtime)
        # the zip may hold the case at its root or inside one folder
        if self._is_case(target):
            return target
        depth = len(self._marker()) - 1
        return next((p.parents[depth] for p in sorted(target.rglob(self._marker()[-1]))
                     if self._is_case(p.parents[depth])), target)

    # --- the parameter file -----------------------------------------------------------
    def _code_texts(self, module_dir: Path) -> list[str]:
        names = list(self._files()) + list((self._section("inputs").get("code_hints") or {}).get("files") or [])
        return [read_text(module_dir / n) for n in dict.fromkeys(names) if (module_dir / n).is_file()]

    def find_parameter_files(self, module_dir: Path) -> None:
        """Step: search for the parameter file. Runs once per module. With
        `inputs.search` set in the software description and no file of the configured
        name present, the best file found (see software/param_files.py)
        replaces the configured name -- as the input file and, when the two
        are the same file, as the case marker too."""
        if getattr(self, "_searched_in", None) == module_dir:
            return
        self._searched_in = module_dir
        self._found_marker = None
        self._input_overrides: dict[str, Any] = {}
        self.parameter_files: list[param_files.FoundFile] = []
        inputs = self._section("inputs")
        search = inputs.get("search")
        if not search or not hasattr(self._format, "sniff") or self._case_files_pattern():
            return
        configured = [str(f) for f in inputs.get("files") or []]
        self.parameter_files = param_files.find_parameter_files(
            module_dir, self._format, search, self._code_texts(module_dir), configured)
        if not self.parameter_files or self._is_case(module_dir) or self._case_dirs_under(module_dir):
            return  # the configured file is there: nothing to replace
        best = self.parameter_files[0].path
        self._input_overrides = {"files": [best.name]}
        if Path(self._section("cases").get("marker", "")).name in configured:
            self._found_marker = (best.name,)

    def _linked_inputs(self) -> dict[str, Any]:
        """The linked file as the input file, when the description names
        none and the input format can read that file; with `cases.files`,
        each case's copy of its file."""
        if self._case_files_pattern():
            return {"files": [self._case_file_name()]}
        inputs = self._section("inputs")
        if self.main_file is None or inputs.get("files") or not hasattr(self._format, "sniff"):
            return {}
        return {"files": [self.main_file.name]} if self._format.sniff(self.main_file) else {}

    def input_settings(self, module_dir: Path) -> dict[str, Any]:
        """The `inputs:` settings, with a searched-for file name applied."""
        self.find_parameter_files(module_dir)
        return {**self._section("inputs"), **self._linked_inputs(), **self._input_overrides}

    def discover_cases(self, module_dir: Path) -> list[tuple[Path, str]]:
        if self._case_files_pattern():
            return self._file_cases(module_dir)
        self.find_parameter_files(module_dir)
        settings = self._section("cases")
        if not settings.get("nested") and self._is_case(module_dir):
            return [(module_dir, module_dir.name)]
        cases = []
        for case_dir in self._case_dirs_under(module_dir):
            if settings.get("ids", "path") == "folder_name":
                case_id = case_dir.name
            else:
                rel = case_dir.relative_to(module_dir)
                case_id = module_dir.name if str(rel) == "." else "_".join(rel.parts)
            cases.append((case_dir, case_id))
        cases += [(self._extract_zip(z, module_dir), z.stem) for z in self._zips_under(module_dir)]
        if not cases:
            sys.exit(f"Error: no {self.name} case (a folder with {settings['marker']}) found under {module_dir}")
        seen: dict[str, Path] = {}
        for case_dir, case_id in cases:
            if case_id in seen:
                sys.exit(f"Duplicate case id '{case_id}' derived from two different directories:\n"
                         f"  {seen[case_id]}\n  {case_dir}")
            seen[case_id] = case_dir
        return cases

    def template_case(self, module_dir: Path) -> Path:
        """The case whose inputs are discovered: the implementation folder
        itself if it is a case, else the first case found."""
        self.find_parameter_files(module_dir)
        if self._is_case(module_dir):
            return module_dir
        return self.discover_cases(module_dir)[0][0]

    # --- the implementation --------------------------------------------------------
    @classmethod
    def _files(cls) -> list[str]:
        return list(cls._section("implementation").get("files") or [])

    @classmethod
    def _has_files(cls, folder: Path) -> bool:
        return all((folder / f).exists() for f in cls._files())

    @classmethod
    def _folders_with_files(cls, root: Path) -> list[Path]:
        files = cls._files()
        return sorted({p.parent for p in root.rglob(files[0]) if cls._has_files(p.parent)})

    @classmethod
    def detect(cls, root: Path) -> bool:
        if cls._files():
            return cls._has_files(root) or any(cls._has_files(p.parent) for p in root.rglob(cls._files()[0]))
        probe = cls()
        if probe._case_files_pattern():
            return bool(probe.case_file_paths(root)) or any(root.rglob(probe._case_files_pattern()))
        return probe._is_case(root) or bool(probe._case_dirs_under(root)) or bool(probe._zips_under(root))

    def find_module_dir(self, root: Path, args: Any) -> Path:
        # a link to a folder or file inside the repository (see
        # repo_source.split_link): a file is the implementation's main
        # file, in its folder; a folder is searched like the repository
        inner = getattr(args, "implementation_path", None)
        if inner:
            path = root / inner
            if path.is_file():
                self.main_file = path
                return path.parent
            if not path.is_dir():
                sys.exit(f"Error: {inner} does not exist in the repository (looked in {root}).")
            root = path
        folder = self._section("implementation").get("folder")
        if folder:
            path = root / folder
            if not path.is_dir():
                sys.exit(f"Error: implementation.folder {folder} does not exist under {root}")
            return path
        if self._case_files_pattern() and not self._files():
            if not self.case_file_paths(root):
                sys.exit(f"Error: no case file ({self._case_files_pattern()}) in {self.case_files_folder(root)} "
                         "-- check cases.files and cases.folder in the software description.")
            return root
        files = self._files()
        if files:
            if self._has_files(root):
                return root
            candidates = self._folders_with_files(root)
            what = " + ".join(files)
            hint = "Set implementation.folder in the software config, or point"
            if not candidates:
                sys.exit(f"Error: could not find a benchmark module (a directory containing {what}) anywhere "
                         f"under {root}.\n{hint} module_dir directly at it, or double-check the path.")
            if len(candidates) > 1:
                listing = "\n  ".join(str(c) for c in candidates)
                sys.exit(f"Error: found multiple benchmark modules ({what}) under {root}:\n  {listing}\n"
                         f"{hint} module_dir directly at the one you want.")
            return candidates[0]

        # the implementation is the folder holding the cases
        if self._is_case(root):
            return root
        cases, zips = self._case_dirs_under(root), self._zips_under(root)
        if len(cases) == 1 and not zips:
            return cases[0]
        homes = sorted({p.parent for p in cases} | {z.parent for z in zips})
        if len(homes) == 1:
            return homes[0]
        if not homes:
            sys.exit(f"Error: no {self.name} case (a folder with {self._section('cases')['marker']}) "
                     f"found under {root}.")
        listing = "\n  ".join(str(h) for h in homes)
        sys.exit(f"Error: found {self.name} cases in several places under {root}:\n  {listing}\n"
                 "Point module_dir at the one you want.")

    def source_texts(self, module_dir: Path) -> dict[str, str]:
        wanted = self.spec.get("sources") or []
        if isinstance(wanted, dict):
            names = set(wanted.get("input_files_named") or [])
            case_dir = self.template_case(module_dir)
            return {str(p.relative_to(case_dir)): read_text(p)
                    for p in self._format.input_files(case_dir, self.input_settings(module_dir))
                    if p.name in names}
        return {name: read_text(module_dir / name) for name in wanted}

    # --- inputs ---------------------------------------------------------------------
    def discover_parameters(self, module_dir: Path) -> list[ParameterCandidate]:
        """Step: identify the parameters in the parameter file, each with
        its location in that file (ParameterCandidate.location)."""
        settings = self.input_settings(module_dir)
        template = self.template_case(module_dir)
        candidates = self._format.discover(template, settings)
        if self._case_files_pattern():
            # one case per file: the location names the file itself
            # (../parameters_1.json), not its copy in .benchmantic/cases/
            original = next((p for d, p in ((d, self.case_sources.get(cid)) for d, cid in
                             self.discover_cases(module_dir)) if d == template and p), None)
            if original is not None:
                shown = os.path.relpath(original, module_dir)
                for c in candidates:
                    if c.location and c.location.file == self._case_file_name():
                        c.location.file = shown
        hints = settings.get("code_hints")
        if hints:
            texts = [read_text(module_dir / name) for name in hints.get("files") or []]
            sources.attach_code_hints(candidates, texts, hints)
        return candidates

    def default_scenario_candidates(self, candidates: list[ParameterCandidate]) -> list[ParameterCandidate]:
        selection = self._section("inputs").get("default_selection") or {}
        if selection.get("placeholders_first"):
            placeholders = [c for c in candidates
                            if c.value.strip().startswith("{") and c.value.strip().endswith("}")]
            if placeholders:
                return placeholders
        patterns = [str(s).lower() for s in selection.get("sections") or []]
        chosen = [c for c in candidates
                  if any(fnmatch.fnmatchcase(c.section.lower(), p) for p in patterns)
                  and (not selection.get("numeric_only")
                       or all(re.match(r"^[-+.\d eE]+$", t) for t in c.tokens))]
        return chosen or list(candidates)

    def resolve_case_params(self, case_dir: Path, parameter_fields: dict[str, Any]) -> dict[str, Any]:
        settings = {**self._section("inputs"), **self._linked_inputs(), **getattr(self, "_input_overrides", {})}
        return self._format.resolve(case_dir, parameter_fields, settings)

    def input_display(self, section: str, key: str) -> str:
        return self.spec.get("display", "{section}:{key}").format(section=section, key=key)

    # --- metrics -----------------------------------------------------------------------
    def metric_files(self, module_dir: Path, repo_root: Path) -> list[Path]:
        return sources.metric_files(module_dir, repo_root, self._section("metrics"))

    def _code_files(self, module_dir: Path, repo_root: Path) -> list[Path]:
        """The code computing the metrics, for `metrics: {from: variables}`:
        `metrics.files` when given, else the input files (or the linked file)."""
        if self._section("metrics").get("files"):
            return self.metric_files(module_dir, repo_root)
        case_dir = self.template_case(module_dir)
        files = self._format.input_files(case_dir, self.input_settings(module_dir))
        return files or ([self.main_file] if self.main_file else [])

    def metric_expressions(self, module_dir: Path, repo_root: Path) -> dict[str, str]:
        """{metric key: Python expression} for metrics computed by the code
        (`metrics: {from: variables}`): `metrics.values` when the description
        lists them, else what python_code.result_variables() finds."""
        settings = self._section("metrics")
        if settings.get("from") != "variables":
            return {}
        if settings.get("values"):
            return {str(k): str(v) for k, v in settings["values"].items()}
        from software.formats import python_code
        found: dict[str, str] = {}
        for path in self._code_files(module_dir, repo_root):
            for key, expression, _ in python_code.result_variables(python_code.read_code(path)):
                found.setdefault(key, expression)
        return found

    def _variable_metrics(self, module_dir: Path, repo_root: Path) -> list[MetricCandidate]:
        from software.formats import python_code
        codes = [python_code.read_code(p) for p in self._code_files(module_dir, repo_root)]
        lines = {k: line for code in codes for k, _, line in python_code.result_variables(code)}
        candidates = []
        for key, expression in self.metric_expressions(module_dir, repo_root).items():
            try:
                context = "\n".join(python_code.expression_context(code, expression, lines.get(key)) for code in codes)
            except SyntaxError:
                sys.exit(f"Error: metric {key}: {expression!r} is not a Python expression.")
            candidates.append(MetricCandidate(key=key, context=context[:600]))
        if not candidates:
            where = ", ".join(str(p) for p in self._code_files(module_dir, repo_root)) or "(no code file)"
            sys.exit(f"No metrics found in {where}: no printed values and no arrays filled in a loop. "
                     "List them in the software description as metrics.values, e.g. "
                     "{drag_coefficient_max: 'max(C_D)'}.")
        return candidates

    def discover_metrics(self, module_dir: Path, repo_root: Path) -> list[MetricCandidate]:
        if self._section("metrics").get("from") == "variables":
            return self._variable_metrics(module_dir, repo_root)
        found: list[MetricCandidate] = []
        seen: set[str] = set()
        files = self.metric_files(module_dir, repo_root)
        for path in files:
            text = read_text(path)
            batch = discover_metrics_in_python(text) if path.suffix == ".py" else discover_metrics(text)
            for c in batch:
                if c.key not in seen:
                    seen.add(c.key)
                    found.append(c)
        if not found:
            where = ", ".join(str(p) for p in files) or "(no file matching metrics.files)"
            sys.exit(f"No metrics found in {where}: expected the JSON keys written to the results "
                     f"(e.g. out << \"  \\\"some_key\\\": \" << value; in C++, or "
                     f"json.dump({{'l2_error': ...}}, ...) into {METRICS_FILE_NAME} in Python).")
        return found

    # --- descriptive metadata -------------------------------------------------------------
    def benchmark_description(self, module_dir: Path, texts: dict[str, str]) -> str:
        how = self.spec.get("description", "markdown_section")
        if how == "doxygen":
            return sources.doxygen_description(list(texts.values()))
        if how == "main_file" and self.main_file is not None:
            found = sources.code_docs_description(self.main_file)
            if found:
                return found
        root = getattr(self, "docs_root", None) or self._repo_root or module_dir
        return sources.markdown_section_description(module_dir, root)

    def benchmark_label(self, module_dir: Path, repo_root: Path, texts: dict[str, str]) -> str | None:
        self._repo_root = repo_root
        how = self.spec.get("label", "markdown_title")
        if how == "cpp_class":
            return sources.cpp_class_label(list(texts.values()))
        if how == "main_file" and self.main_file is not None:
            found = sources.code_docs_label(self.main_file)
            if found:
                return found
        return sources.markdown_title_label(module_dir, repo_root)

    def executable_name(self, module_dir: Path, repo_root: Path) -> tuple[str | None, Path | None]:
        self._repo_root = repo_root
        settings = self._section("executable")
        if settings.get("from") == "cmake":
            return sources.cmake_executable(module_dir, repo_root, list(settings.get("test_macros") or []),
                                            settings.get("source"))
        if settings.get("from") == "entry":
            path = self.template_case(module_dir) / settings["file"]
            value = self._format.read_entry(path, settings["key"])
            return (value, path) if value else (None, None)
        if self.run_settings.get("type") == "parameter_file":   # the last program file the command runs
            names = re.findall(r"[\w./-]+\.(?:py|jl|sh|R|m)\b", str(self.run_settings.get("command", "")))
            found = [module_dir / n for n in names if (module_dir / n).is_file()]
            return (found[-1].name, found[-1]) if found else (None, None)
        if self.run_settings.get("type") == "script":          # the script itself
            script = self.script_file(module_dir)
            return (script.name, script) if script else (None, None)
        return None, None

    def script_file(self, module_dir: Path) -> Path | None:
        """For `run.type: script`: `run.script`, else the linked file, else
        the first input file."""
        named = self.run_settings.get("script")
        if named:
            return module_dir / named
        if self.main_file is not None:
            return self.main_file
        files = self._format.input_files(self.template_case(module_dir), self.input_settings(module_dir))
        return files[0] if files else None

    def build_hints(self, module_dir: Path, repo_root: Path) -> dict[str, Any]:
        return self.style().build_hints(self, module_dir, repo_root)

    # --- workflow ---------------------------------------------------------------------------
    def render_snakefile(self, ctx: SnakefileContext) -> SnakefileResult:
        return self.style().render(self, ctx)

    # --- LLM context ---------------------------------------------------------------------------
    def prompt_context(self) -> str:
        return str(self.spec.get("notes") or "").strip()
