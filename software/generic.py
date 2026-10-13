# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.generic

The one adapter behind every software.yaml. A software folder only says
*what* its implementation looks like (which files mark it, where the cases
are, the input format, where metrics are written, how a case is run); the
steps themselves are the same for every code and live here, in
software/sources.py, software/formats/ and snakefile/styles/.

A folder may add a hooks.py for something no setting covers. It can define
  add_snakefile_arguments(group)              extra Snakefile options
  render_snakefile(adapter, ctx, style)       replaces the style's render()
"""

from __future__ import annotations

import fnmatch
import importlib.util
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


def load_spec(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        sys.exit(f"Error: reading {path} needs PyYAML (pip install pyyaml).")
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(spec, dict) or not spec.get("name"):
        sys.exit(f"Error: {path} must be a YAML mapping with at least a `name`.")
    return spec


def _load_hooks(folder: Path) -> ModuleType | None:
    path = folder / "hooks.py"
    if not path.is_file():
        return None
    module_spec = importlib.util.spec_from_file_location(f"software_hooks_{folder.name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class GenericAdapter(SoftwareAdapter):
    spec: dict[str, Any] = {}
    hooks: ModuleType | None = None
    _folder: Path | None = None
    #: Repository root last seen (label/executable); a fallback for the docs.
    _repo_root: Path | None = None

    @classmethod
    def for_folder(cls, folder: Path) -> type[GenericAdapter]:
        """An adapter class for software/<folder>/software.yaml."""
        spec = load_spec(folder / "software.yaml")
        selection = (spec.get("inputs") or {}).get("default_selection") or {}
        attrs = {
            "spec": spec,
            "hooks": _load_hooks(folder),
            "_folder": folder,
            "name": spec["name"],
            "slug": folder.name,
            "input_label": spec.get("input_label", f"the {spec['name']} input files"),
            "default_selection_label": selection.get("label", "all parameters"),
            "default_container_shared_dir": (spec.get("snakefile") or {}).get("container_shared_dir"),
        }
        class_name = re.sub(r"\W", "", spec["name"]) + "Adapter"
        return type(class_name, (cls,), attrs)

    # --- settings ---------------------------------------------------------------
    @classmethod
    def folder(cls) -> Path:
        return cls._folder

    @classmethod
    def _section(cls, key: str) -> dict[str, Any]:
        return cls.spec.get(key) or {}

    @property
    def snakefile_settings(self) -> dict[str, Any]:
        return self._section("snakefile")

    @classmethod
    def style(cls) -> ModuleType:
        return get_style(cls._section("snakefile").get("style", "command_line"))

    @property
    def _format(self) -> ModuleType:
        return get_format(self._section("inputs").get("format", "ini"))

    # --- cases ---------------------------------------------------------------------
    def _marker(self) -> tuple[str, ...]:
        """The file marking a case: the configured one, or the parameter
        file found by search when the configured one isn't there."""
        return getattr(self, "_found_marker", None) or Path(self._section("cases")["marker"]).parts

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
        `inputs.search` set in software.yaml and no file of the configured
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
        if not search or not hasattr(self._format, "sniff"):
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

    def input_settings(self, module_dir: Path) -> dict[str, Any]:
        """The `inputs:` settings, with a searched-for file name applied."""
        self.find_parameter_files(module_dir)
        return {**self._section("inputs"), **self._input_overrides}

    def discover_cases(self, module_dir: Path) -> list[tuple[Path, str]]:
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
        return probe._is_case(root) or bool(probe._case_dirs_under(root)) or bool(probe._zips_under(root))

    @classmethod
    def _pick_option(cls) -> str | None:
        return cls._section("implementation").get("pick_option")

    def find_module_dir(self, root: Path, args: Any) -> Path:
        files = self._files()
        if files:
            option = self._pick_option()
            picked = getattr(args, option.replace("-", "_"), None) if option else None
            if picked is not None:
                if not Path(picked).exists():
                    sys.exit(f"--{option} {picked} does not exist")
                return Path(picked).parent
            if self._has_files(root):
                return root
            candidates = self._folders_with_files(root)
            what = " + ".join(files)
            hint = f"Pass --{option} to pick one, or point" if option else "Point"
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
        candidates = self._format.discover(self.template_case(module_dir), settings)
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
        settings = {**self._section("inputs"), **getattr(self, "_input_overrides", {})}
        return self._format.resolve(case_dir, parameter_fields, settings)

    def input_display(self, section: str, key: str) -> str:
        return self.spec.get("display", "{section}:{key}").format(section=section, key=key)

    # --- metrics -----------------------------------------------------------------------
    def metric_files(self, module_dir: Path, repo_root: Path) -> list[Path]:
        return sources.metric_files(module_dir, repo_root, self._section("metrics"))

    def discover_metrics(self, module_dir: Path, repo_root: Path) -> list[MetricCandidate]:
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
        root = getattr(self, "docs_root", None) or self._repo_root or module_dir
        return sources.markdown_section_description(module_dir, root)

    def benchmark_label(self, module_dir: Path, repo_root: Path, texts: dict[str, str]) -> str | None:
        self._repo_root = repo_root
        if self.spec.get("label", "markdown_title") == "cpp_class":
            return sources.cpp_class_label(list(texts.values()))
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
        return None, None

    def build_hints(self, module_dir: Path, repo_root: Path) -> dict[str, Any]:
        return self.style().build_hints(self, module_dir, repo_root)

    # --- command-line options and workflow -------------------------------------------------
    @classmethod
    def add_arguments(cls, group) -> None:
        option = cls._pick_option()
        if option:
            files = cls._files()
            group.add_argument(f"--{option}", type=Path, default=None, dest=option.replace("-", "_"),
                               help=f"Explicit path to {files[0]}, only needed if more than one "
                                    f"{'+'.join(files)} pair is found under module_dir.")

    @classmethod
    def add_snakefile_arguments(cls, group) -> None:
        if cls.hooks and hasattr(cls.hooks, "add_snakefile_arguments"):
            cls.hooks.add_snakefile_arguments(group)

    def render_snakefile(self, ctx: SnakefileContext) -> SnakefileResult:
        if self.hooks and hasattr(self.hooks, "render_snakefile"):
            return self.hooks.render_snakefile(self, ctx, self.style())
        return self.style().render(self, ctx)

    # --- LLM context ---------------------------------------------------------------------------
    def prompt_context(self) -> str:
        return str(self.spec.get("llm_notes") or "").strip()
