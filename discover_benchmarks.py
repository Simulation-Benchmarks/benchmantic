#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
discover_benchmarks.py

Compares the entries of the DuMux benchmark list with the repositories of a
GitHub account, lists the entries that have no repository yet, and prints, as
JSON, those to create -- every new entry, or only those given with --create --
for the GitHub Actions workflow discover-benchmarks.yml, which creates one
repository per entry.

    python3 discover_benchmarks.py --org Sarbani-Roy
    python3 discover_benchmarks.py --org Sarbani-Roy --create "Taylor-Couette Flow"

An entry's repository is named after its title ("Taylor-Couette Flow" ->
taylor-couette-flow). An entry can only be turned into a repository once it has
a run config, examples/<name>/benchmantic.yaml (the implementation must write
the metrics benchmantic reads; that is not automated); new entries without one
are listed but not created. Needs the GitHub CLI
(gh) unless --repos is given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DUMUX_BENCHMARKS = "https://dumux.org/docs/doxygen/master/benchmarks.html"
#: the DuMux list marks each benchmark's link with this arrow ("↗️ Taylor-Couette Flow")
ARROW = "↗"


class _Links(HTMLParser):
    """Every link of the page: [section heading, link text, href]."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[list[str]] = []
        self.section = ""
        self._heading: list[str] | None = None
        self._link: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3"):
            self._heading = []
        elif tag == "a" and self._link is None:
            self._link = [self.section, "", dict(attrs).get("href") or ""]

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3") and self._heading is not None:
            self.section = " ".join("".join(self._heading).split())
            self._heading = None
        elif tag == "a" and self._link is not None:
            self.links.append(self._link)
            self._link = None

    def handle_data(self, data):
        if self._heading is not None:
            self._heading.append(data)
        if self._link is not None:
            self._link[1] += data


def slug(title: str) -> str:
    """The repository name of an entry: "Taylor-Couette Flow" -> taylor-couette-flow."""
    return re.sub(r"[^0-9a-z]+", "-", title.lower()).strip("-")


def parse_entries(html: str, base: str = DUMUX_BENCHMARKS) -> list[dict]:
    """The benchmarks of the DuMux list: title, repository name, section, link."""
    parser = _Links()
    parser.feed(html)
    entries, seen = [], set()
    for section, text, href in parser.links:
        text = " ".join(text.split())
        if not text.startswith(ARROW):
            continue
        title = text.lstrip(ARROW + "️ ").strip()
        name = slug(title)
        if name and name not in seen:
            seen.add(name)
            entries.append(dict(title=title, name=name, section=section,
                                link=urllib.request.urljoin(base, href)))
    return entries


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "benchmantic"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def repositories(org: str) -> set[str]:
    """The names of the repositories of a GitHub account (lower case)."""
    proc = subprocess.run(["gh", "repo", "list", org, "--limit", "1000", "--json", "name", "--jq", ".[].name"],
                          capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"Error: could not list the repositories of {org}: {proc.stderr.strip()}")
    return {line.strip().lower() for line in proc.stdout.splitlines() if line.strip()}


def summary(lines: list[str]) -> None:
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", required=True, help="GitHub account whose repositories count as existing.")
    ap.add_argument("--url", default=DUMUX_BENCHMARKS, help="The benchmark list (default: %(default)s).")
    ap.add_argument("--html", type=Path, help="Read the list from this file instead of --url.")
    ap.add_argument("--repos", help="Comma-separated repository names instead of asking GitHub (for testing).")
    ap.add_argument("--create", action="append", default=[], metavar="TITLE",
                    help="Create (or update) only the entry with this title (repeatable), whether or not it has a "
                         "repository; the others are only listed. Default: every new entry with a run config.")
    ap.add_argument("--examples", type=Path, default=ROOT / "examples", help="Where run configs are (default: examples/).")
    args = ap.parse_args(argv)

    entries = parse_entries(args.html.read_text(encoding="utf-8") if args.html else fetch(args.url), args.url)
    if not entries:
        sys.exit(f"Error: found no benchmarks in {args.url} -- has the page changed?")
    chosen = {slug(t) for t in args.create}
    unknown = chosen - {e["name"] for e in entries}
    if unknown:
        sys.exit(f"Error: not in the list: {', '.join(sorted(unknown))}; it has: "
                 f"{', '.join(e['title'] for e in entries)}")
    existing = {r.strip().lower() for r in args.repos.split(",")} if args.repos is not None else repositories(args.org)

    for e in entries:
        e["exists"] = e["name"] in existing
        config = args.examples / e["name"] / "benchmantic.yaml"
        e["config"] = str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else str(config)
        e["has_config"] = config.is_file()
    new = [e for e in entries if not e["exists"]]
    todo = [e for e in entries if e["name"] in chosen] if chosen else [e for e in new if e["has_config"]]
    for e in todo:
        if not e["has_config"]:
            sys.exit(f"Error: {e['title']} has no run config {e['config']}")

    def state(e):
        if e in todo:
            return "**update**" if e["exists"] else "**create**"
        return "listed only" if e["has_config"] or chosen else f"needs `{e['config']}`"
    summary([f"### DuMux benchmarks: {len(entries)} in the list, {len(new)} without a repository in "
             f"`{args.org}`, {len(todo)} to create or update", "",
             "| Benchmark | Section | Repository | |", "|---|---|---|---|"]
            + [f"| [{e['title']}]({e['link']}) | {e['section']} | `{e['name']}` | {state(e)} |"
               for e in new + [e for e in todo if e["exists"]]] + [""])
    print(f"{len(new)} of {len(entries)} DuMux benchmarks have no repository in {args.org}:", file=sys.stderr)
    for e in new:
        print(f"  {e['title']} -> {e['name']}", file=sys.stderr)
    for e in todo:
        print(f"{'update' if e['exists'] else 'create'}: {e['title']} -> {args.org}/{e['name']}", file=sys.stderr)
    print(json.dumps(dict(entries=todo, new=[e["name"] for e in new],
                          existing=[e["name"] for e in entries if e["exists"]])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
