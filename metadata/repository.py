# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.repository

Software-neutral scraping of the source repository: README/AUTHORS/
CITATION.cff discovery and parsing, SPDX license/copyright headers, and
repo-URL-derived publisher/author guesses. How a code's own sources are
read (module layout, build files, cases) lives in software/<name>/.
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path


# REUSE-IgnoreStart
# (the two patterns below contain literal "SPDX-License-Identifier:" /
# "SPDX-FileCopyrightText:" substrings used to *parse* other files' headers,
# not an actual header of this file -- reuse-tool's naive text scan would
# otherwise misdetect them and try to parse what follows as this file's own
# SPDX License Expression / copyright line.)
SPDX_LICENSE_PATTERN = re.compile(r"SPDX-License-Identifier:\s*([^\s*]+)")
SPDX_COPYRIGHT_PATTERN = re.compile(r"SPDX-FileCopyrightText:\s*(?:Copyright\s*(?:©|\(c\))?\s*)?([^\n*]+)")


def extract_spdx_info(*source_texts: str) -> dict[str, str | None]:
    """Scan for SPDX-License-Identifier / SPDX-FileCopyrightText header
    comments (standard in REUSE-compliant sources) and turn them into
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


def find_repo_file(names: tuple[str, ...], module_dir: Path, repo_root: Path) -> Path | None:
    """Look for one of `names` starting in the benchmark module folder,
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
    """Fallback benchmark description source for when the software sources
    carry no description: use the first real paragraph of the
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




def read_citation_cff(path: Path | None) -> dict:
    """Authors, license and repository URL from a CITATION.cff file
    (https://citation-file-format.github.io). Returns {} when there's no
    file, it can't be parsed, or PyYAML isn't installed.

    authors: [{"name", "orcid", "affiliation"}, ...] (people and entities)
    """
    if path is None or not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 -- optional metadata source
        return {}
    if not isinstance(data, dict):
        return {}
    authors = []
    for a in data.get("authors") or []:
        if not isinstance(a, dict):
            continue
        name = " ".join(str(a[k]) for k in ("given-names", "name-particle", "family-names") if a.get(k)) \
            or str(a.get("name") or "").strip()
        if name:
            authors.append({"name": name, "orcid": a.get("orcid"), "affiliation": a.get("affiliation")})
    license_id = data.get("license")
    if isinstance(license_id, list):
        license_id = license_id[0] if license_id else None
    return {
        "authors": authors,
        "license": license_id,
        "repository_code": data.get("repository-code") or data.get("repository"),
        "version": data.get("version"),
        "title": data.get("title"),
    }
