#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
create_benchmark_repo.py

Turns benchmantic's outputs into a benchmark catalog repository laid out like
https://github.com/Simulation-Benchmarks/rotating-cylinders, and optionally
creates it on GitHub and waits for its GitHub Actions.

    python3 workflow.py --config examples/taylor-couette-flow/benchmantic.yaml
    python3 create_benchmark_repo.py --config examples/taylor-couette-flow/benchmantic.yaml \\
        [--org Sarbani-Roy] [--github --watch]

The repository is repository_template/ with its @@PLACEHOLDERS@@ filled in from
the benchmark description (outputs/<software-name>/*_benchmark.jsonld) and the
`repository:` block of the run config, plus the Snakefile, the container
folder, CITATION.cff, codemeta.json and the license texts. --github needs the
GitHub CLI (gh) signed in with a token that may create repositories in --org
and push workflow files (classic token: scopes repo + workflow).

--rohub (with --github) also sets up RoHub: it finds or creates the benchmark's
research object on the development RoHub (research area: `rohub_research_area`,
else the one the LLM chooses from RoHub's list) (and on production with
`rohub_production: true` or a production login), stores the RoHub logins and
the research objects' identifiers in the new repository, and starts its RoHub
Integration CI. The logins come from the environment: ROHUB_DEV_USERNAME,
ROHUB_DEV_PASSWORD (and ROHUB_PROD_USERNAME, ROHUB_PROD_PASSWORD).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from metadata import repo_source  # noqa: E402
from snakefile.renderer import DEFAULT_UNIT_SYMBOLS  # noqa: E402

TEMPLATE = ROOT / "repository_template"
#: benchmantic's own files next to an implementation: not part of the container
NOT_IN_CONTAINER = (".*_cache.json", ".benchmantic*", "__pycache__", ".git")


# --- the benchmark description ------------------------------------------------------

def _ids(value) -> list[str]:
    items = value if isinstance(value, list) else [value] if value else []
    return [i["@id"] if isinstance(i, dict) else i for i in items]


def _types(node: dict) -> list:
    t = node.get("@type", [])
    return t if isinstance(t, list) else [t]


def read_description(path: Path) -> dict:
    """Name, description, software, parameters, metrics and configurations."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_id = {n.get("@id"): n for n in doc["@graph"]}
    bench = next(n for n in doc["@graph"] if "m4i:Benchmark" in _types(n))
    unit = lambda node: (node.get("has unit") or {}).get("@id")  # noqa: E731
    text = lambda node: node.get("dcterms:description") or node.get("description") or ""  # noqa: E731

    parameters, configurations = {}, []
    for set_id in _ids(bench.get("has parameter set")):
        pset, values = by_id[set_id], {}
        for var in (by_id[i] for i in _ids(pset.get("has part"))):
            values[var["label"]] = var.get("has numerical value", var.get("has string value"))
            parameters.setdefault(var["label"], (unit(var), text(var)))
        configurations.append((pset.get("identifier") or set_id, values))
    metrics = [(by_id[i]["label"], unit(by_id[i]), text(by_id[i])) for i in _ids(bench.get("evaluates"))]
    software = next((n.get("label") for n in doc["@graph"] if "tool" in _types(n)), None) or "simulation"
    # the implementation's doc comment, without its Doxygen commands
    description = "\n".join(re.sub(r"^\\brief\s*", "", line.strip()) for line in text(bench).splitlines()
                            if not re.match(r"\s*\\(file|ingroup)\b", line))
    return dict(label=bench.get("label", path.stem), description=description.strip(), software=software,
                version=bench.get("schema:version") or "1.0.0", parameters=parameters, metrics=metrics,
                configurations=configurations)


def _unit(u: str | None) -> str:
    fragment = re.split(r"[:/#]", u or "")[-1]
    return "-" if fragment.upper() in ("", "UNITLESS") else fragment


def tables(desc: dict) -> dict[str, str]:
    """Markdown for the README and the documentation."""
    confs = desc["configurations"]
    varying = [p for p in desc["parameters"]
               if len(confs) < 2 or len({json.dumps(v.get(p)) for _, v in confs}) > 1]
    rows = ["| Configuration | " + " | ".join(varying) + " |", "|---|" + "---:|" * len(varying)]
    rows += [f"| `{cid}` | " + " | ".join(f"`{v.get(p)}`" for p in varying) + " |" for cid, v in confs]
    first = confs[0][1] if confs else {}
    params = ["| Parameter | Unit | Value | Description |", "|---|---|---|---|"]
    params += [f"| `{p}` | {_unit(u)} | {'see configurations' if p in varying and len(confs) > 1 else f'`{first.get(p)}`'} "
               f"| {d or '-'} |" for p, (u, d) in desc["parameters"].items()]
    metrics = ["| Metric | Unit | Description |", "|---|---|---|"]
    metrics += [f"| `{m}` | {_unit(u)} | {d or '-'} |" for m, u, d in desc["metrics"]]
    return dict(CONFIGURATION_TABLE="\n".join(rows), PARAMETER_TABLE="\n".join(params),
                METRIC_TABLE="\n".join(metrics),
                METRICS_LIST="\n".join(f"- **`{m}`**" + (f": {d}" if d else "") for m, _, d in desc["metrics"]))


def rohub_settings(desc: dict, settings: dict) -> dict[str, str]:
    """RoHub and notebook placeholders: production or development RoHub, the
    parameter the notebook plots the metrics against (`plot_parameter`, else the
    first numeric parameter that differs between configurations) and the metrics."""
    production = bool(settings.get("rohub_production", False))
    confs = [values for _, values in desc["configurations"]]
    numeric = [p for p in desc["parameters"]
               if all(isinstance(v.get(p), (int, float)) and not isinstance(v.get(p), bool) for v in confs)]
    varying = [p for p in numeric if len({v.get(p) for v in confs}) > 1]
    plot = settings.get("plot_parameter") or (varying or numeric or list(desc["parameters"]) or ["configuration"])[0]
    if plot not in desc["parameters"]:
        sys.exit(f"Error: plot_parameter {plot!r} is not a parameter of the benchmark: {', '.join(desc['parameters'])}")
    return dict(ROHUB_PRODUCTION="true" if production else "false", ROHUB_PRODUCTION_PY=str(production),
                PLOT_PARAMETER=plot, METRICS_PY=repr([m for m, _, _ in desc["metrics"]]))


# --- writing the repository ------------------------------------------------------------

def fill(text: str, values: dict) -> str:
    """Replace every @@NAME@@; an unknown name is an error."""
    def sub(m):
        if m.group(1) not in values:
            raise KeyError(f"no value for @@{m.group(1)}@@")
        return str(values[m.group(1)])
    return re.sub(r"@@([A-Z0-9_]+)@@", sub, text)


def parse_author(text: str) -> dict:
    """'Given Family; ORCID; affiliation' (ORCID and affiliation optional)."""
    parts = [p.strip() for p in text.split(";")] + ["", ""]
    *given, family = parts[0].split()
    orcid = parts[1] and (parts[1] if parts[1].startswith("http") else f"https://orcid.org/{parts[1]}")
    return dict(given=" ".join(given), family=family, orcid=orcid, affiliation=parts[2])


def citation(values: dict, authors: list[dict], keywords: list[str]) -> str:
    q = json.dumps
    lines = ["cff-version: 1.2.0", 'message: "If you use this software, please cite it using the metadata from this file."',
             f"title: {q(values['TITLE'])}", f"abstract: {q(values['SUMMARY_ONE_LINE'])}", "type: software", "authors:"]
    for a in authors or [dict(name=f"The {values['REPO_NAME']} authors")]:
        if "name" in a:
            lines.append(f"  - name: {q(a['name'])}")
            continue
        lines += [f"  - family-names: {q(a['family'])}", f"    given-names: {q(a['given'])}"]
        lines += [f"    orcid: {q(a['orcid'])}"] * bool(a["orcid"]) + [f"    affiliation: {q(a['affiliation'])}"] * bool(a["affiliation"])
    lines += ['version: "0.1.0"', f'date-released: "{values["DATE"]}"', "license: MIT",
              f'repository-code: "{values["URL"]}"', f'url: "{values["URL"]}"', "keywords:"]
    lines += [f"  - {q(k)}" for k in keywords]
    lines += ["references:", "  - type: software", f"    title: {q(values['SOFTWARE_NAME'])}",
              f"    url: {q(values['SOFTWARE_URL'])}", "  - type: software", '    title: "benchmantic"',
              '    url: "https://github.com/Simulation-Benchmarks/benchmantic"']
    return "\n".join(lines) + "\n"


def codemeta(values: dict, authors: list[dict], keywords: list[str]) -> str:
    people = [{"@type": "Person", "givenName": a["given"], "familyName": a["family"],
               **({"@id": a["orcid"]} if a["orcid"] else {}),
               **({"affiliation": {"@type": "Organization", "name": a["affiliation"]}} if a["affiliation"] else {})}
              for a in authors]
    return json.dumps({
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0", "@type": "SoftwareSourceCode",
        "name": values["REPO_NAME"], "description": values["SUMMARY_ONE_LINE"], "version": "0.1.0",
        "dateCreated": values["DATE"], "license": "https://spdx.org/licenses/MIT",
        "codeRepository": values["URL"], "issueTracker": f"{values['URL']}/issues",
        "programmingLanguage": {"@type": "ComputerLanguage", "name": "Python"}, "keywords": keywords,
        "author": people or [{"@type": "Organization", "name": f"The {values['REPO_NAME']} authors"}],
    }, indent=2) + "\n"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True,
        stdin: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, input=stdin)
    if check and proc.returncode:
        sys.exit(f"Error: {' '.join(cmd[:3])} ... failed:\n{(proc.stderr or proc.stdout).strip()}")
    return proc


def build(outputs: Path, out: Path, settings: dict, org: str, config_dir: Path) -> None:
    """Write the repository into `out`."""
    found = sorted(outputs.glob("*_benchmark.jsonld"))
    if len(found) != 1 or not (outputs / "Snakefile").is_file():
        sys.exit(f"Error: expected one *_benchmark.jsonld and a Snakefile in {outputs}, found "
                 f"{[p.name for p in found]} -- run workflow.py first, and remove older outputs.")
    benchmark = found[0]
    desc = read_description(benchmark)
    snakefile = (outputs / "Snakefile").read_text(encoding="utf-8")
    shared = re.search(r"^container_shared_dir = '(.*)'", snakefile, re.M)
    name = settings["name"]
    software_dir = re.sub(r"[^0-9a-z]+", "-", desc["software"].lower()).strip("-")
    summary = " ".join((settings.get("summary") or desc["description"].split("\n\n")[0]).split())
    benchmark_rel = f"benchmark/{desc['version']}/minimal-configurations.json"
    authors = [parse_author(a) for a in settings.get("authors", [])]
    if not authors or not authors[0]["orcid"]:
        sys.exit("Error: `repository: authors:` needs a first author with an ORCID "
                 "('Given Family; 0000-0000-0000-0000; affiliation'): the RO-Crate of every run names them.")
    agent = authors[0]
    values = dict(
        TITLE=settings.get("title") or f"{desc['label'].title()} Benchmark", SUMMARY=summary,
        SUMMARY_ONE_LINE=summary.replace('"', "'"), DESCRIPTION=desc["description"] or summary,
        ORG=org, REPO_NAME=name, URL=f"https://github.com/{org}/{name}", DATE=datetime.date.today().isoformat(),
        YEAR=datetime.date.today().year, SOFTWARE_NAME=desc["software"], SOFTWARE_DIR=software_dir,
        SOFTWARE_VERSION=settings.get("software_version", "unknown"), SOFTWARE_URL=settings.get("software_url", ""),
        CONTAINER_SHARED_DIR=shared.group(1) if shared else "/shared", BENCHMARK_FILE=benchmark_rel,
        BENCHMARK_VERSION=desc["version"], ENV_NAME=f"{name}-model-validation",
        LOCAL_IMAGE=f"local/{name}-{software_dir}:ci", N_CONFIGURATIONS=len(desc["configurations"]),
        UNIT_SYMBOLS=json.dumps({re.split(r"[:/#]", u)[-1]: s for u, s in DEFAULT_UNIT_SYMBOLS.items()}),
        AGENT_NAME=f"{agent['given']} {agent['family']}", AGENT_ORCID=agent["orcid"],
        **rohub_settings(desc, settings),
        EXTRA_LICENSES="", **tables(desc))

    if out.exists():
        shutil.rmtree(out)
    # the container: a folder, or a GitHub/GitLab link to one
    container = str(settings.get("container") or "")
    if not container:
        sys.exit("Error: `repository: container:` -- the folder with the Dockerfile -- is missing in the run config.")
    source, ref, inner = repo_source.split_link(container)
    clone = repo_source.resolve_source(source, ref=ref) if repo_source.is_remote_source(source) else None
    folder = (clone / (inner or "")) if clone else (config_dir / container)
    if not (folder / "Dockerfile").is_file():
        sys.exit(f"Error: {container}: no Dockerfile there.")
    shutil.copytree(folder, out / software_dir / "container", ignore=shutil.ignore_patterns(*NOT_IN_CONTAINER))
    if clone:
        repo_source.cleanup_ephemeral_clone(clone)
    # REUSE-IgnoreStart
    licenses = set(re.findall(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)", "".join(
        p.read_text(errors="ignore")[:2000] for p in (out / software_dir / "container").rglob("*") if p.is_file())))
    # REUSE-IgnoreEnd
    extra = sorted(licenses - {"MIT", "CC-BY-4.0", "CC0-1.0"})
    values["EXTRA_LICENSES"] = "".join(f" Files under `{software_dir}/container/` whose header names {lic} are "
                                       f"licensed under [{lic}](./LICENSES/{lic}.txt)." for lic in extra)

    # repository_template/, filled in ("dot.x" -> ".x", "x.template" -> "x")
    for path in TEMPLATE.rglob("*"):
        if path.is_file():
            rel = fill(str(path.relative_to(TEMPLATE)), values).replace("dot.", ".").removesuffix(".template")
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(fill(path.read_text(encoding="utf-8"), values), encoding="utf-8")
    docs = settings.get("documentation")
    if docs:
        shutil.copy(config_dir / docs, out / "docs/benchmark-documentation.md")
    shutil.copy(outputs / "Snakefile", out / software_dir / "Snakefile")
    # the description, with its local: ids pointing at the new repository
    doc = json.loads(benchmark.read_text(encoding="utf-8"))
    if isinstance(doc.get("@context"), dict):
        doc["@context"]["local"] = values["URL"] + "/"
    (out / benchmark_rel).parent.mkdir(parents=True, exist_ok=True)
    (out / benchmark_rel).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    keywords = settings.get("keywords") or ["benchmark", desc["label"]]
    (out / "CITATION.cff").write_text(citation(values, authors, keywords), encoding="utf-8")
    (out / "codemeta.json").write_text(codemeta(values, authors, keywords), encoding="utf-8")

    # license texts and the REUSE check (--root: the new repository, even inside another checkout)
    reuse = shutil.which("reuse") or sys.exit("Error: needs the REUSE tool: pip install reuse")
    run([reuse, "--root", str(out), "download", "--all"], cwd=out)
    lint = run([reuse, "--root", str(out), "lint"], cwd=out, check=False)
    if lint.returncode:
        sys.exit("Error: the repository is not REUSE-compliant:\n" + lint.stdout.split("# SUMMARY")[0])
    print(f"✓ Wrote {out} (REUSE-compliant)")


# --- GitHub ------------------------------------------------------------------------------

def summary(*lines: str) -> None:
    """Append to the GitHub Actions job summary, when running in Actions."""
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def publish(out: Path, full: str, description: str) -> bool:
    """Create the GitHub repository `full` from `out` and push it; if it exists, add
    `out` as one new commit on top of its main branch. False if nothing changed."""
    git = ["git", "-C", str(out)]
    if not run(["git", "config", "user.email"], check=False).stdout.strip():
        git += ["-c", "user.name=benchmantic", "-c", "user.email=benchmantic@users.noreply.github.com"]
    run(git + ["init", "-q", "-b", "main"])
    view = run(["gh", "repo", "view", full, "--json", "url", "--jq", ".url"], check=False)
    if view.returncode == 0:
        run(git + ["remote", "add", "origin", view.stdout.strip()])
        if run(git + ["ls-remote", "--heads", "origin", "main"]).stdout.strip():
            run(git + ["fetch", "-q", "origin", "main"])
            run(git + ["reset", "-q", "--mixed", "FETCH_HEAD"])  # its history, our files
    run(git + ["add", "-A"])
    if run(git + ["diff", "--cached", "--quiet"], check=False).returncode == 0:
        return False
    run(git + ["commit", "-q", "-m", "Benchmark repository generated by benchmantic"])
    if view.returncode == 0:
        run(git + ["push", "-q", "-u", "origin", "HEAD:main"])
    else:
        run(["gh", "repo", "create", full, "--public", "--description", description[:350],
             "--source", str(out), "--remote", "origin", "--push"])
    return True


def watch(out: Path, full: str, expect: tuple[str, ...] = (), since: str = "",
          timeout: float = 7200, poll: float = 30) -> bool:
    """Wait for the Actions runs of the pushed commit (the latest of each workflow,
    and for the workflows in `expect` one started at or after `since`); True if
    all succeeded."""
    sha = run(["git", "-C", str(out), "rev-parse", "HEAD"]).stdout.strip()
    deadline = time.monotonic() + timeout
    while True:
        rows = json.loads(run(["gh", "run", "list", "--repo", full, "--limit", "50", "--json",
                               "workflowName,status,conclusion,url,headSha,createdAt"]).stdout or "[]")
        latest: dict[str, dict] = {}
        for r in rows:  # newest first
            if r["headSha"] == sha:
                latest.setdefault(r["workflowName"], r)
        runs = list(latest.values())
        started = all(e in latest and latest[e]["createdAt"] >= since for e in expect)
        if runs and started and all(r["status"] == "completed" for r in runs):
            break
        if time.monotonic() > deadline:
            sys.exit(f"Error: the Actions runs of {full} did not finish within {timeout:.0f} s.")
        print(f"  waiting for {', '.join(r['workflowName'] for r in runs if r['status'] != 'completed') or 'the runs to start'} ...")
        time.sleep(poll)
    table = ["| Workflow | Result |", "|---|---|"]
    for r in runs:
        ok = r["conclusion"] == "success"
        print(f"  {'✓' if ok else '✗'} {r['workflowName']}: {r['conclusion']}  {r['url']}")
        table.append(f"| {r['workflowName']} | {'✅' if ok else '❌'} [{r['conclusion']}]({r['url']}) |")
    summary(*table, "")
    return all(r["conclusion"] == "success" for r in runs)


# --- RoHub ------------------------------------------------------------------------------------

#: research areas for a new research object, in order of preference (a word in the area's name)
RESEARCH_AREAS = ("fluid", "mechanical engineering", "mechanics", "engineering", "physics")


def ai_research_area(areas: list[str], benchmark: dict, provider: str | None = None) -> str | None:
    """The research area the LLM chooses for the benchmark from RoHub's `areas`;
    None if no LLM can be asked (no key, no answer from the list)."""
    if not areas:
        return None
    from ai.inference import _query_llm_json_array, choose_provider

    def check(data):
        if not (isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict)):
            raise ValueError("Expected a JSON array with exactly one object.")
        answer = str(data[0].get("research_area", "")).strip()
        match = next((a for a in areas if a.lower() == answer.lower()), None)
        if not match:
            raise ValueError(f"{answer!r} is not one of the allowed research areas; copy one exactly.")
        data[0]["research_area"] = match
        return data

    system = ("You assign scientific research objects to a research area of RoHub. Answer with a JSON array "
              'holding exactly one object: {"research_area": "<one area, copied exactly from the list>", '
              '"reason": "<one short sentence>"}. Choose the most specific area that fits the subject.')
    prompt = "\n".join([
        f"Benchmark: {benchmark['label']}", f"Simulation software: {benchmark['software']}",
        f"Description: {benchmark['description'][:3000]}",
        f"Parameters: {', '.join(benchmark['parameters'])}",
        f"Metrics: {', '.join(m for m, _, _ in benchmark['metrics'])}", "",
        "Allowed research areas (choose exactly one):", *(f"- {a}" for a in sorted(areas))])
    try:
        answer = _query_llm_json_array(
            provider=choose_provider(provider), model=None, system_prompt=system, prompt=prompt,
            item_count=1, item_label="research area", validator=check, retries=3, retry_delay=2.0,
            verbose=False)[0]
    except (Exception, SystemExit) as e:  # no key, no network, no valid answer: fall back
        print(f"  ! the LLM could not choose a research area ({str(e).splitlines()[0][:200]}); using the default rule")
        return None
    print(f"  the LLM chose the research area {answer['research_area']!r}: {answer.get('reason', '')}")
    return answer["research_area"]


def research_area(areas: list[str], wanted: str | None = None, benchmark: dict | None = None,
                  provider: str | None = None) -> str:
    """`wanted` (any capitalisation; must be one of RoHub's `areas`), else the one
    the LLM chooses for `benchmark`, else the first area matching RESEARCH_AREAS,
    else the first area."""
    if wanted:
        match = next((a for a in areas if a.lower() == wanted.lower()), None)
        if areas and not match:
            sys.exit(f"Error: rohub_research_area {wanted!r} is not a RoHub research area; "
                     f"these are: {', '.join(sorted(areas))}")
        return match or wanted
    chosen = benchmark and ai_research_area(areas, benchmark, provider)
    if chosen:
        return chosen
    for word in RESEARCH_AREAS:
        match = next((a for a in sorted(areas) if word in a.lower()), None)
        if match:
            return match
    return areas[0] if areas else "Engineering"

def research_object(name: str, description: str, url: str, jsonld: Path, production: bool,
                    login: tuple[str, str], area: str | None = None, llm_provider: str | None = None,
                    known: str = "") -> str:
    """The UUID of the benchmark's research object on this RoHub: `known`, else the
    one of this login titled `name`, else a new one with the benchmark description
    and the repository's URL. Its research area is set to `area`, else the one the
    LLM chooses (see research_area()) -- also on an existing research object."""
    try:
        import rohub
        from semantic_benchmark.rohub import login_to_rohub
    except ImportError:
        sys.exit("Error: --rohub needs semantic-benchmark: pip install -r requirements.txt")
    where = "production" if production else "development"
    try:
        login_to_rohub(*login, use_production_rohub=production)
        if not known:
            mine = rohub.list_my_ros()
            if not mine.empty and (mine["title"] == name).any():
                known = str(mine[mine["title"] == name].iloc[0]["identifier"])
        area = research_area(rohub.list_valid_research_areas() or [], area, read_description(jsonld),
                             os.environ.get("LLM_PROVIDER") or llm_provider)
        if known:
            ro = rohub.ros_load(known)
            current = [a.get("name", a) if isinstance(a, dict) else a for a in (ro.research_areas or [])]
            if current != [area]:
                try:
                    ro.research_areas = [area]
                    ro.update()
                    print(f"  research area of {known} on the {where} RoHub: {', '.join(map(str, current)) or '-'} -> {area}")
                except (Exception, SystemExit) as e:  # not worth failing the run for
                    print(f"  ! could not change the research area of {known} to {area}: {e}")
            return known
        print(f"  creating the research object on the {where} RoHub (research area: {area})")
        ro = rohub.ros_create(title=name, research_areas=[area], description=description[:2000])
        ro.add_annotations(body_specification_json=[{"property": "https://schema.org/codeRepository", "value": url}])
        types = rohub.list_valid_resource_types() or []
        if "Annotation Collection" in types:  # what download-semantic-benchmark looks for
            ro.add_internal_resource(res_type="Annotation Collection", file_path=str(jsonld),
                                     title="benchmark.json", description="Semantic description of the benchmark")
        else:
            print(f"  ! the {where} RoHub has no resource type 'Annotation Collection': the benchmark "
                  "description was not added, so 'Download benchmark resources from RoHub' will not work")
        return str(ro.identifier)
    except SystemExit as e:  # the rohub client exits on errors
        sys.exit(f"Error: {where} RoHub: {e}")


def setup_rohub(full: str, url: str, out: Path, settings: dict, llm_provider: str | None = None) -> list[str]:
    """Find or create the research objects, store the logins and identifiers in
    `full`; returns the RoHub instances set up."""
    production = bool(settings.get("rohub_production", False))
    jsonld = next((out / "benchmark").glob("*/minimal-configurations.json"))
    done = []
    for kind in ["DEV"] + ["PROD"] * (production or bool(os.environ.get("ROHUB_PROD_USERNAME"))):
        login = os.environ.get(f"ROHUB_{kind}_USERNAME", ""), os.environ.get(f"ROHUB_{kind}_PASSWORD", "")
        if not all(login):
            sys.exit(f"Error: --rohub needs the RoHub login in ROHUB_{kind}_USERNAME and ROHUB_{kind}_PASSWORD.")
        variable = f"ROHUB_{kind}_BENCHMARK_IDENTIFIER"
        stored = run(["gh", "variable", "get", variable, "--repo", full], check=False)
        known = (str(settings.get(f"rohub_{kind.lower()}_identifier") or "")
                 or (stored.stdout.strip() if stored.returncode == 0 else ""))
        uuid = research_object(settings["name"], settings.get("summary") or settings["name"], url, jsonld,
                               kind == "PROD", login, settings.get("rohub_research_area"), llm_provider, known)
        for secret, value in zip(("USERNAME", "PASSWORD"), login):
            run(["gh", "secret", "set", f"ROHUB_{kind}_{secret}", "--repo", full], stdin=value)
        run(["gh", "variable", "set", variable, "--repo", full, "--body", uuid])
        where = "production" if kind == "PROD" else "development"
        print(f"✓ RoHub ({where}): research object {uuid}")
        summary(f"RoHub ({where}): research object `{uuid}`", "")
        done.append(where)
    return done


def start_rohub_run(full: str) -> str:
    """Start the new repository's RoHub Integration CI; returns when (UTC)."""
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for _ in range(18):  # GitHub needs a moment to see a newly pushed workflow
        if run(["gh", "workflow", "run", "rohub-integration.yml", "--repo", full, "--ref", "main"],
               check=False).returncode == 0:
            print("✓ Started its RoHub Integration CI")
            return since
        time.sleep(10)
    sys.exit(f"Error: could not start the RoHub Integration CI of {full}.")


# --- command line ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="The run config: its `repository:` block and `software-name` (where workflow.py wrote the outputs).")
    ap.add_argument("--outputs", type=Path, help="benchmantic's outputs (default: outputs/<software-name>).")
    ap.add_argument("--output", type=Path, help="Where to write the repository (default: outputs/repositories/<name>).")
    ap.add_argument("--name", help="Repository name (default: `name` in the `repository:` block).")
    ap.add_argument("--org", default="Simulation-Benchmarks", help="GitHub account or organization (default: %(default)s).")
    ap.add_argument("--github", action="store_true", help="Create the repository on GitHub (or update it) and push.")
    ap.add_argument("--watch", action="store_true", help="With --github: wait for its Actions; fail unless all pass.")
    ap.add_argument("--rohub", action="store_true",
                    help="With --github: set up RoHub (see above) and start the repository's RoHub Integration CI.")
    args = ap.parse_args(argv)
    if (args.watch or args.rohub) and not args.github:
        ap.error("--watch and --rohub need --github")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    settings = dict(config.get("repository") or {})
    settings["name"] = args.name or settings.get("name")
    if not settings["name"]:
        sys.exit("Error: no repository name: `name` in the `repository:` block, or --name.")
    software = config.get("software-name") or config.get("software_name") or str(config.get("software", "")).lower()
    outputs = args.outputs or Path("outputs") / software
    out = (args.output or Path("outputs") / "repositories" / settings["name"]).resolve()
    build(outputs, out, settings, args.org, args.config.resolve().parent)
    if not args.github:
        return 0

    full = f"{args.org}/{settings['name']}"
    url = f"https://github.com/{full}"
    pushed = publish(out, full, settings.get("summary") or settings["name"])
    print(f"✓ Pushed {url}" if pushed else f"✓ {url} already holds exactly these files")
    summary(f"### Benchmark repository: [{full}]({url})", "")
    expect, since = (), ""
    if args.rohub:
        setup_rohub(full, url, out, settings, config.get("llm_provider"))
        since, expect = start_rohub_run(full), ("RoHub Integration CI",)
    if args.watch:
        print(f"  Waiting for its GitHub Actions: {url}/actions")
        return 0 if watch(out, full, expect, since) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
