"""
metadata.repository

Everything about locating and scraping the source repository itself:
resolving the benchmark module directory inside a larger checkout,
CMakeLists.txt executable-target detection, README/AUTHORS discovery and
parsing, SPDX license/copyright headers, C++ class-name detection, Doxygen
benchmark description extraction, case (params.input) discovery, and
repo-URL-derived publisher/author guesses.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

from utils import camel_to_label, read_text

SPDX_LICENSE_PATTERN = re.compile(r"SPDX-License-Identifier:\s*([^\s*]+)")
SPDX_COPYRIGHT_PATTERN = re.compile(r"SPDX-FileCopyrightText:\s*(?:Copyright\s*(?:©|\(c\))?\s*)?([^\n*]+)")
CLASS_NAME_PATTERN = re.compile(r"\bclass\s+([A-Z]\w+)\s*(?::\s*public|\{)")


def extract_spdx_info(*source_texts: str) -> dict[str, str | None]:
    """Scan for SPDX-License-Identifier / SPDX-FileCopyrightText header
    comments (standard in DuMux/DUNE source files) and turn them into
    RO-Crate license/publisher fields. Returns {} entries as None if not
    found in any of the given texts.
    """
    info: dict[str, str | None] = {"license_id": None, "copyright": None}
    for text in source_texts:
        if not text:
            continue
        if info["license_id"] is None:
            m = SPDX_LICENSE_PATTERN.search(text)
            if m:
                info["license_id"] = m.group(1).strip()
        if info["copyright"] is None:
            m = SPDX_COPYRIGHT_PATTERN.search(text)
            if m:
                copyright_text = m.group(1).strip().rstrip(".")
                # Trim common trailing pointer clauses, e.g. "..., see
                # AUTHORS.md in root folder" -> "..." -- we want just the
                # copyright holder name, not the whole sentence.
                copyright_text = re.split(r",?\s+see\s+\S+", copyright_text, maxsplit=1)[0].strip()
                info["copyright"] = copyright_text
    return info



def extract_class_label(*source_texts: str) -> str | None:
    for text in source_texts:
        if not text:
            continue
        m = CLASS_NAME_PATTERN.search(text)
        if m:
            return camel_to_label(m.group(1))
    return None



DOXYGEN_BLOCK_PATTERN = re.compile(r"/\*!(.*?)\*/", re.DOTALL)


def extract_benchmark_description(*source_texts: str) -> str:
    """Return the longest Doxygen-style /*! ... */ comment block found
    across the given source texts, with the leading '*' decoration and any
    leading \\brief tag stripped. Returns "" if none is found.
    """
    best = ""
    for text in source_texts:
        if not text:
            continue
        for block in DOXYGEN_BLOCK_PATTERN.findall(text):
            cleaned = re.sub(r"^[ \t]*\*[ \t]?", "", block, flags=re.MULTILINE).strip()
            cleaned = re.sub(r"^\\brief\s*", "", cleaned)
            if len(cleaned) > len(best):
                best = cleaned
    return best



def find_module_dir(root: Path, main_cc_override: Path | None) -> Path:
    """Resolve the directory that actually holds the benchmark's main.cc +
    problem.hh, so the `module_dir` CLI argument can point at a top-level
    repo checkout (e.g. .../dumux_test/rotating-cylinders) instead of
    requiring the exact leaf directory
    (.../test/freeflow/navierstokes/rotatingcylinders).

    Search order:
      1. --main-cc was given explicitly -> use its parent directory.
      2. `root` itself already contains both main.cc and problem.hh -> use it.
      3. Otherwise, recursively search under `root` for a main.cc that has a
         problem.hh sitting next to it (the pairing is what distinguishes
         "the benchmark module" from an unrelated main.cc elsewhere in a
         larger checkout, e.g. other DuMux examples/tests).
    """
    if main_cc_override is not None:
        if not main_cc_override.exists():
            sys.exit(f"--main-cc {main_cc_override} does not exist")
        return main_cc_override.parent

    if (root / "main.cc").exists() and (root / "problem.hh").exists():
        return root

    candidates = sorted({
        p.parent for p in root.rglob("main.cc")
        if (p.parent / "problem.hh").exists()
    })

    if not candidates:
        sys.exit(
            f"Error: could not find a benchmark module (a directory containing "
            f"both main.cc and problem.hh) anywhere under {root}.\n"
            "Pass --main-cc to point directly at the right main.cc, or double-check the path."
        )
    if len(candidates) > 1:
        listing = "\n  ".join(str(c) for c in candidates)
        sys.exit(
            f"Error: found multiple benchmark modules (main.cc + problem.hh pairs) under {root}:\n"
            f"  {listing}\n"
            "Pass --main-cc to pick one, or point module_dir directly at the one you want."
        )
    return candidates[0]


#: Matches CMake's literal add_executable(<target> <source1> <source2> ...).
#: Less common in DuMux/DUNE test suites, which normally go through the
#: dumux_add_test/dune_add_test macro below -- checked as a fallback.
ADD_EXECUTABLE_PATTERN = re.compile(r"add_executable\s*\(\s*([\w.\-]+)\s+([^)]*)\)", re.IGNORECASE | re.DOTALL)

#: Matches DuMux/DUNE's dumux_add_test(...)/dune_add_test(...) CMake macros
#: -- the standard way DuMux/DUNE test executables are actually declared,
#: e.g.:
#:   dumux_add_test(NAME test_ff_navierstokes_rotatingcylinders
#:                   SOURCES main.cc
#:                   COMPILE_DEFINITIONS ...)
#: These take CMake keyword arguments (NAME/SOURCES/...) in any order, so
#: the whole parenthesized block is captured and searched rather than
#: assuming a fixed argument order.
ADD_TEST_MACRO_PATTERN = re.compile(r"(?:dumux|dune)_add_test\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)
NAME_ARG_PATTERN = re.compile(r"\bNAME\s+([\w.\-]+)", re.IGNORECASE)


def extract_executable_name(cmakelists_text: str) -> str | None:
    """Scan one CMakeLists.txt's text for the compiled executable target
    built from main.cc -- this is the exact binary name a Snakemake
    workflow needs to invoke, and it's not always guessable from the
    directory name alone (e.g. DuMux abbreviates 'freeflow' to 'ff':
    test/freeflow/navierstokes/rotatingcylinders builds
    test_ff_navierstokes_rotatingcylinders, not
    test_freeflow_navierstokes_rotatingcylinders).

    Tries dumux_add_test/dune_add_test (the actual DuMux/DUNE convention)
    first, then a literal add_executable(...) as a fallback. Within each,
    prefers a call block that explicitly references main.cc; if none does
    (e.g. the source list is a CMake variable rather than a literal
    filename), falls back to the first target name found of that kind.
    Returns None if neither pattern matches at all.
    """
    if not cmakelists_text:
        return None

    first_test_macro_name = None
    for m in ADD_TEST_MACRO_PATTERN.finditer(cmakelists_text):
        block = m.group(1)
        name_match = NAME_ARG_PATTERN.search(block)
        if not name_match:
            continue
        if first_test_macro_name is None:
            first_test_macro_name = name_match.group(1)
        if re.search(r"\bmain\.cc\b", block, re.IGNORECASE):
            return name_match.group(1)
    if first_test_macro_name:
        return first_test_macro_name

    first_executable_target = None
    for m in ADD_EXECUTABLE_PATTERN.finditer(cmakelists_text):
        target, sources = m.group(1), m.group(2)
        if first_executable_target is None:
            first_executable_target = target
        if re.search(r"\bmain\.cc\b", sources, re.IGNORECASE):
            return target
    return first_executable_target


def find_executable_name(module_dir: Path, repo_root: Path) -> tuple[str | None, Path | None]:
    """Find the compiled executable's name by scanning CMakeLists.txt
    file(s), trying (in order):
      1. Right next to main.cc/problem.hh, then walking up toward
         repo_root (the normal DuMux/DUNE layout -- see find_repo_file()).
      2. If that file doesn't exist, or exists but extract_executable_name()
         doesn't find a match in it (e.g. an unrecognized macro), fall back
         to traversing every CMakeLists.txt anywhere under repo_root and
         using the first one that does match.
    Returns (executable_name, path_it_was_found_in), both None if nothing
    in the whole repo matches.
    """
    direct = find_repo_file(("CMakeLists.txt",), module_dir, repo_root)
    if direct:
        name = extract_executable_name(read_text(direct))
        if name:
            return name, direct

    for candidate in sorted(repo_root.rglob("CMakeLists.txt")):
        if candidate == direct:
            continue
        name = extract_executable_name(read_text(candidate))
        if name:
            return name, candidate

    return None, None


def find_repo_file(names: tuple[str, ...], module_dir: Path, repo_root: Path) -> Path | None:
    """Look for one of `names` starting right next to main.cc/problem.hh,
    then walking up toward `repo_root` (covers repos where a file like
    README.md/AUTHORS.md lives at the top level rather than inside the
    specific test-case folder). Returns None if nothing is found by the
    time we reach `repo_root`.
    """
    directory = module_dir
    while True:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
        if directory == repo_root or directory == directory.parent:
            break
        directory = directory.parent
    return None


#: Filenames checked when looking for a module's README, in preference order.
README_FILENAMES: tuple[str, ...] = ("README.md", "Readme.md", "readme.md", "README", "README.rst", "README.txt")

#: Filenames checked when looking for a module's author/contributor list.
AUTHORS_FILENAMES: tuple[str, ...] = (
    "AUTHORS.md", "AUTHORS", "AUTHORS.txt",
    "CONTRIBUTORS.md", "CONTRIBUTORS", "CONTRIBUTORS.txt",
)


def find_readme(module_dir: Path, repo_root: Path) -> Path | None:
    return find_repo_file(README_FILENAMES, module_dir, repo_root)


#: 'SPDX-FileCopyrightText: ..., see AUTHORS.md in root folder' -- a direct
#: pointer, when present, to exactly which file holds the real author list.
SPDX_SEE_FILE_PATTERN = re.compile(r"SPDX-FileCopyrightText:.*?,?\s+see\s+(\S+)", re.IGNORECASE)


def extract_authors_file_hint(*source_texts: str) -> str | None:
    """Look for a 'SPDX-FileCopyrightText: ..., see AUTHORS.md in root
    folder' style pointer in the source and return the referenced filename
    (e.g. 'AUTHORS.md'), if found. This lets find_authors_file() check that
    exact name first, ahead of the generic AUTHORS_FILENAMES guess list.
    """
    for text in source_texts:
        if not text:
            continue
        m = SPDX_SEE_FILE_PATTERN.search(text)
        if m:
            return m.group(1).rstrip(".,;:")
    return None


def find_authors_file(module_dir: Path, repo_root: Path, hint: str | None = None) -> Path | None:
    """Find the module's author/contributor list file. Checks the SPDX-
    comment-hinted filename first (if any), then falls back to the common
    AUTHORS_FILENAMES, searching from module_dir up to repo_root.
    """
    names = ((hint,) if hint else ()) + AUTHORS_FILENAMES
    return find_repo_file(names, module_dir, repo_root)


def extract_authors_list(authors_text: str, limit: int = 15) -> tuple[list[str], int]:
    """Parse a plain-text/markdown AUTHORS-style file into a list of names
    (trailing '<email@...>' addresses stripped), skipping headings, blank
    lines, and comment lines. Returns (names[:limit], total_count_found) so
    callers can note how many entries were omitted from the crate.
    """
    if not authors_text:
        return [], 0
    names: list[str] = []
    for line in authors_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        stripped = re.sub(r"^[*\-+]\s+", "", stripped)
        stripped = re.sub(r"\s*<[^<>]+>\s*$", "", stripped).strip()
        if stripped:
            names.append(stripped)
    return names[:limit], len(names)


def extract_readme_description(readme_text: str) -> str:
    """Fallback benchmark description source for when problem.hh/main.cc
    have no Doxygen \\brief block: use the first real paragraph of the
    module's README, skipping headings, blank lines, and badge/image-only
    lines. Leading bullet markers are stripped so a bullet list folded into
    the paragraph reads as plain prose rather than showing stray '*'/'-'.
    """
    if not readme_text:
        return ""
    paragraph: list[str] = []
    for line in readme_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("![") or stripped.startswith("["):
            continue
        stripped = re.sub(r"^[*\-+]\s+", "", stripped)
        paragraph.append(stripped)
    return " ".join(paragraph).strip()


#: REUSE-convention (https://reuse.software) license link, e.g.
#: '[GPL-3.0-or-later.txt](LICENSES/GPL-3.0-or-later.txt)' -- the filename
#: itself is the SPDX identifier, which is far more reliable than parsing
#: free-text license prose.
README_LICENSE_LINK_PATTERN = re.compile(r"LICENSES/([\w.+-]+?)\.(?:txt|md)", re.IGNORECASE)

#: 'git clone <url> [folder]' from install instructions.
README_GIT_CLONE_PATTERN = re.compile(r"git clone\s+(\S+)")


def extract_readme_license(readme_text: str) -> str | None:
    """Pull an SPDX license identifier out of a REUSE-style license link in
    the README (e.g. a 'LICENSES/GPL-3.0-or-later.txt' path), if present.
    Returns None if no such link is found.
    """
    if not readme_text:
        return None
    m = README_LICENSE_LINK_PATTERN.search(readme_text)
    return m.group(1) if m else None


def extract_readme_repo_url(readme_text: str) -> str | None:
    """Pull the repo's clone URL out of a 'git clone <url> ...' snippet in
    the README's installation instructions, if present. Returns None if no
    such snippet is found.
    """
    if not readme_text:
        return None
    m = README_GIT_CLONE_PATTERN.search(readme_text)
    return m.group(1) if m else None


def extract_readme_dependencies(readme_text: str) -> list[dict[str, str]]:
    """Parse a DUNE/DuMux-style 'Version Information' markdown table, e.g.

        | module name | branch name | commit sha | commit date |
        |-------------|-------------|------------|--------------|
        | dune-istl   | ...         | ...        | ...          |

    into a list of {"module", "branch", "commit", "date"} dicts (pinned
    dependency versions this benchmark was built/tested against). Returns
    [] if no such table is found. Column matching is by header name (not
    position), so column order/count in the source table doesn't matter.
    """
    if not readme_text:
        return []

    def split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    lines = readme_text.splitlines()
    deps: list[dict[str, str]] = []
    header_cells: list[str] | None = None

    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            if header_cells is not None:
                break  # table ended
            continue

        cells = split_row(line)

        if header_cells is None:
            lower_cells = [c.lower() for c in cells]
            if any("module" in c for c in lower_cells) and any("commit" in c for c in lower_cells):
                header_cells = lower_cells
            continue

        # Separator row right after the header, e.g. "|---|:---:|---|"
        if re.fullmatch(r":?-{2,}:?", cells[0]):
            continue

        if len(cells) != len(header_cells):
            break

        row = dict(zip(header_cells, cells))
        deps.append({
            "module": next((v for k, v in row.items() if "module" in k), "").strip("` "),
            "branch": next((v for k, v in row.items() if "branch" in k), ""),
            "commit": next((v for k, v in row.items() if "commit sha" in k or "sha" in k), ""),
            "date": next((v for k, v in row.items() if "date" in k), ""),
        })

    return deps


#: Public code-hosting domains where the domain itself isn't an
#: organization -- for these, the meaningful "who owns this repo" signal is
#: the first path segment (the GitHub/GitLab.com/etc. org or user
#: namespace), not the shared hosting domain.
GENERIC_CODE_HOSTS: set[str] = {
    "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "codeberg.org", "git.sr.ht",
}


def _domain_to_label(domain: str) -> str:
    """'iws.uni-stuttgart' -> 'Iws Uni Stuttgart'. A coarse, best-effort
    label -- not an authoritative institution name.
    """
    words = domain.replace("-", " ").replace(".", " ").split()
    return " ".join(w.capitalize() for w in words)


def extract_publisher_from_repo_url(repo_url: str | None) -> dict[str, str] | None:
    """Best-effort organization guess from a repo's clone URL, used only as
    a last-resort fallback (see build_manifest()) when no SPDX copyright
    header is present anywhere in the source. Returns {"name", "url"} or
    None if nothing sensible can be derived.

    - On a self-hosted-looking domain, the domain itself is treated as the
      institution, e.g. 'git.iws.uni-stuttgart.de' -> 'Iws Uni Stuttgart'
      (stripping a leading 'git.'/'www.' prefix first).
    - On a well-known public code forge (github.com, gitlab.com, ...) the
      domain says nothing about who owns the repo, so the first path
      segment -- the org/user namespace -- is used instead, e.g.
      github.com/SomeOrg/some-repo -> 'SomeOrg'.

    This is a guess, not a verified fact -- callers should flag it as such
    (see build_manifest()'s use of "disambiguating_note").
    """
    if not repo_url:
        return None
    parsed = urllib.parse.urlparse(repo_url)
    host = (parsed.hostname or "").lower()
    if not host:
        return None

    if host in GENERIC_CODE_HOSTS:
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            return None
        org = parts[0]
        return {"name": org, "url": f"https://{host}/{org}"}

    stripped = re.sub(r"^(git|www|gitlab|source)\.", "", host)
    domain_body = stripped.rsplit(".", 1)[0] if "." in stripped else stripped
    label = _domain_to_label(domain_body)
    if not label:
        return None
    return {"name": label, "url": f"https://{host}"}


# =============================================================================
# 5. Execution Orchestration
# =============================================================================



def case_id_for(params_path: Path, root: Path) -> str:
    rel_dir = params_path.parent.relative_to(root)
    if str(rel_dir) == ".":
        return params_path.parent.name
    return "_".join(rel_dir.parts)


def discover_cases(root: Path) -> list[tuple[Path, str]]:
    found = sorted(root.rglob("params.input"))
    if not found:
        sys.exit(f"No params.input files found anywhere under {root}")

    cases: list[tuple[Path, str]] = []
    seen_ids: dict[str, Path] = {}
    for params_path in found:
        case_dir = params_path.parent
        case_id = case_id_for(params_path, root)
        if case_id in seen_ids:
            sys.exit(
                f"Duplicate case id '{case_id}' derived from two different directories:\n"
                f"  {seen_ids[case_id]}\n  {case_dir}"
            )
        seen_ids[case_id] = case_dir
        cases.append((case_dir, case_id))
    return cases


