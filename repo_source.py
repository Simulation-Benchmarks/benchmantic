#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
repo_source.py

Lets `module_dir` (the positional argument to describe_benchmark.py /
metadata.builder) be a GitHub, GitLab, or any git-reachable URL instead of
only a local folder path.

By default, resolve_source() clones a remote source into a throwaway temp
directory and the caller deletes it (cleanup_ephemeral_clone()) once the
run is done -- nothing is left behind in the repo, matching the plain "run
it against a URL" use case where you don't want a
.benchmantic_repo_cache/ directory accumulating on disk. Pass --keep-clone
to opt into the old behavior instead: a deterministic, persistent local
cache directory keyed by the URL (and --ref, if given), reused (fetched +
checked out in place, not re-cloned) on the next run against the same
URL -- along with the .parameter_metadata_cache.json /
.metric_metadata_cache.json living inside it (see ai.cache), so repeat
runs against the same remote benchmark don't re-query the LLM for
everything every time. --clone-dir (an explicit destination) always
implies --keep-clone, since choosing a specific location only makes sense
if you intend to come back to it.

A local path is returned completely unchanged -- this module is a no-op
for the existing local-folder workflow, and cleanup_ephemeral_clone() is a
no-op for it too (it only ever removes directories it created itself,
under EPHEMERAL_CLONE_ROOT).

Usage
-----
    python3 describe_benchmark.py https://github.com/org/dumux-benchmarks \\
        --ref main --container-image ... --container-shared-dir ...

    python3 describe_benchmark.py git@gitlab.com:group/project.git \\
        --keep-clone --container-image ... --container-shared-dir ...

    python3 describe_benchmark.py /local/path/to/checkout \\
        --container-image ... --container-shared-dir ...
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

#: Persistent cache location used only with --keep-clone (or --clone-dir).
#: Override with BENCHMANTIC_REPO_CACHE, or per-invocation with --clone-dir.
DEFAULT_CLONE_ROOT = SCRIPT_DIR / ".benchmantic_repo_cache"

#: Root for throwaway clones (the default, no --keep-clone/--clone-dir).
#: Kept as a known, dedicated subtree of the OS temp dir specifically so
#: cleanup_ephemeral_clone() can safely verify "this is a directory we
#: created" before ever calling shutil.rmtree() on it.
EPHEMERAL_CLONE_ROOT = Path(tempfile.gettempdir()) / "benchmantic_ephemeral_clones"

#: Matches any "scheme://..." URL (https://github.com/..., git://..., ssh://...).
_URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
#: Matches an scp-like git remote, e.g. "git@github.com:org/repo.git".
_SCP_LIKE_PATTERN = re.compile(r"^[\w.\-]+@[\w.\-]+:")


def is_remote_source(module_dir: str) -> bool:
    """True if `module_dir` looks like something git can clone (a URL, or
    an scp-like git@host:path spec) rather than a local filesystem path.
    Deliberately conservative -- a bare local path never matches either
    pattern, including one that happens to contain a colon on Windows
    (`C:\\...`), since neither pattern matches a single-letter scheme.
    """
    return bool(_URL_SCHEME_PATTERN.match(module_dir) or _SCP_LIKE_PATTERN.match(module_dir))


def clone_root() -> Path:
    env = os.environ.get("BENCHMANTIC_REPO_CACHE")
    return Path(env) if env else DEFAULT_CLONE_ROOT


def _slugify_source(url: str, ref: str | None) -> str:
    """Turn a repo URL (+ optional ref) into a filesystem-safe directory
    name, e.g. 'https://github.com/Org/repo.git' + 'main' ->
    'github.com_Org_repo__main'.
    """
    slug = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", url)  # strip scheme
    slug = re.sub(r"^[\w.\-]+@", "", slug)  # strip scp-like user@
    slug = slug.rstrip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", slug).strip("_")
    if ref:
        slug += f"__{re.sub(r'[^A-Za-z0-9]+', '_', ref).strip('_')}"
    return slug or "repo"


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(
            f"Error: git {' '.join(args)} failed (exit {result.returncode}):\n"
            f"{(result.stderr or result.stdout).strip()}"
        )


def _clone(url: str, dest: Path, ref: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> Cloning {url} into {dest}...")
    args = ["clone", "--depth", "1"]
    if ref:
        args += ["--branch", ref]
    args += [url, str(dest)]
    _run_git(args)


def _update(dest: Path, ref: str | None) -> None:
    print(f"-> Updating existing clone at {dest}...")
    # "HEAD" when no ref was given -- explicitly re-fetches the remote's
    # current default branch tip, rather than relying on FETCH_HEAD landing
    # on the right branch out of whatever a plain `git fetch` pulls in.
    refspec = ref or "HEAD"
    _run_git(["fetch", "--depth", "1", "origin", refspec], cwd=dest)
    _run_git(["checkout", "-q", "FETCH_HEAD"], cwd=dest)


def resolve_source(
    module_dir: str,
    *,
    ref: str | None = None,
    clone_dir: Path | None = None,
    fresh_clone: bool = False,
    keep_clone: bool = False,
) -> Path:
    """If `module_dir` is a local path, return it unchanged (as a Path).

    If it's a GitHub/GitLab/any git URL, clone it (or update an existing
    clone) and return the local path -- everything downstream
    (metadata.builder, review.py's cache files, discover_cases(), etc.)
    then works exactly as if a local checkout had been passed in the first
    place.

    Where it's cloned to depends on `keep_clone`/`clone_dir`:
      - `clone_dir` given: always used as-is, fetched+checked-out in place
        on a later call against the same URL (implies persistence -- an
        explicit destination is a signal you intend to reuse it).
      - `keep_clone=True`, no `clone_dir`: a deterministic path under
        clone_root(), same fetch-in-place reuse as above.
      - neither (the default): a fresh clone into a throwaway directory
        under EPHEMERAL_CLONE_ROOT. The caller is expected to remove it
        with cleanup_ephemeral_clone() once it's done with it -- this
        function does not, and does not track it beyond returning the path.
    """
    if not is_remote_source(module_dir):
        return Path(module_dir)

    if clone_dir is None and not keep_clone:
        EPHEMERAL_CLONE_ROOT.mkdir(parents=True, exist_ok=True)
        dest = Path(tempfile.mkdtemp(
            prefix=f"{_slugify_source(module_dir, ref)}_", dir=EPHEMERAL_CLONE_ROOT,
        ))
        # mkdtemp() already created `dest` as an empty directory -- git is
        # fine cloning into an existing empty one, so no fresh_clone/exists
        # branching needed here, unlike the persistent case below.
        _clone(module_dir, dest, ref)
        return dest

    dest = clone_dir or (clone_root() / _slugify_source(module_dir, ref))

    if dest.exists() and fresh_clone:
        shutil.rmtree(dest)

    if dest.exists():
        if not (dest / ".git").is_dir():
            sys.exit(
                f"Error: {dest} already exists but isn't a git checkout -- remove it, "
                "pass --clone-dir to use a different location, or --fresh-clone to overwrite it."
            )
        _update(dest, ref)
    else:
        _clone(module_dir, dest, ref)

    return dest


def is_ephemeral_clone(path: Path) -> bool:
    """True if `path` is a clone resolve_source() created under
    EPHEMERAL_CLONE_ROOT itself -- i.e. safe for cleanup_ephemeral_clone()
    to remove. False for a local (non-cloned) module_dir, and false for a
    persistent clone (--keep-clone / --clone-dir), so cleanup never touches
    either of those.
    """
    try:
        path.resolve().relative_to(EPHEMERAL_CLONE_ROOT.resolve())
        return True
    except (ValueError, OSError):
        return False


def cleanup_ephemeral_clone(path: Path) -> None:
    """Remove `path` if (and only if) it's a throwaway clone
    resolve_source() created -- a no-op for a local module_dir or a
    persistent (--keep-clone/--clone-dir) clone. Errors during removal are
    swallowed (best-effort tidy-up, not something worth failing the whole
    run over) but reported, in case a file lock or permission issue leaves
    something behind for the user to clean up by hand.
    """
    if not is_ephemeral_clone(path):
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        print(f"warning: could not remove temporary clone {path}: {exc}", file=sys.stderr)
