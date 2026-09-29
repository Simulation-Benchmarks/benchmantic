# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.mardi

Looks up a benchmark's research problem and mathematical model on the MaRDI
portal (a Wikibase), so the description can link to real entities instead
of local placeholder ids.

How it works:
  1. Search the portal for items that are instances of "research problem"
     (P31 = Q6534292), matching a few search terms taken from the source
     (the benchmark name, and any parenthetical in the doc-comment such as
     "(Taylor-Couette flow)"). Labels AND aliases are matched -- e.g. the
     Taylor-Couette flow item carries the alias "rotating cylinders
     problem". The type filter matters: an unfiltered search returns
     thousands of zbMATH papers with the same words in their title.
  2. For each candidate, follow P1513 (research problem -> mathematical
     model) to get the model, and fetch English labels/descriptions for
     both, using MaRDI's own spelling (e.g. with an en dash).

Only the MaRDI Wikibase API is used (action=query&list=search with
CirrusSearch's haswbstatement filter, and action=wbgetentities). Network
errors raise MardiError so callers can fall back to placeholders.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Callable

API_URL = "https://portal.mardi4nfdi.de/w/api.php"
ENTITY_URL = "https://portal.mardi4nfdi.de/entity/{qid}"
ITEM_NAMESPACE = 120

INSTANCE_OF = "P31"
RESEARCH_PROBLEM_CLASS = "Q6534292"
MATHEMATICAL_MODEL_CLASS = "Q68663"
#: research problem -> mathematical model that models it
PROBLEM_TO_MODEL = "P1513"

USER_AGENT = "benchmantic (https://github.com/Simulation-Benchmarks/benchmantic)"
TIMEOUT_SECONDS = 15
MAX_CANDIDATES = 5

FetchJson = Callable[[dict[str, Any]], dict[str, Any]]


class MardiError(RuntimeError):
    """The portal couldn't be reached or returned something unexpected."""


def _fetch_json(params: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_URL}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- any network/parse problem means "no lookup"
        raise MardiError(f"MaRDI portal request failed: {exc}") from exc


def entity_url(qid: str) -> str:
    return ENTITY_URL.format(qid=qid)


def search_terms(benchmark_label: str | None, description: str | None) -> list[str]:
    """Search terms, most specific first: parentheticals from the
    doc-comment (e.g. 'Taylor-Couette flow'), then the benchmark name.
    Skips years '(1923)' and fragments like '(Navier-)'."""
    terms: list[str] = []
    for match in re.finditer(r"\(([^()]{4,80})\)", description or ""):
        term = " ".join(match.group(1).split())
        if term.endswith("-") or re.fullmatch(r"[\d\s,.;-]+", term):
            continue
        terms.append(term)
    if benchmark_label:
        terms.append(benchmark_label)
    seen: set[str] = set()
    return [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]


def search_items(term: str, class_qid: str, fetch_json: FetchJson = _fetch_json) -> list[str]:
    """QIDs of items that are instances of class_qid and match term."""
    data = fetch_json({
        "action": "query",
        "list": "search",
        "srsearch": f"{term} haswbstatement:{INSTANCE_OF}={class_qid}",
        "srnamespace": ITEM_NAMESPACE,
        "srlimit": MAX_CANDIDATES,
    })
    if "query" not in data:
        raise MardiError(f"unexpected search response: {str(data)[:200]}")
    return [hit["title"].split(":", 1)[-1] for hit in data["query"].get("search", [])]


def get_entities(qids: list[str], fetch_json: FetchJson = _fetch_json) -> dict[str, dict[str, Any]]:
    """qid -> {qid, label, description, url, claims} (English)."""
    if not qids:
        return {}
    data = fetch_json({
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|descriptions|claims",
        "languages": "en",
    })
    result = {}
    for qid, entity in (data.get("entities") or {}).items():
        if "missing" in entity:
            continue
        result[qid] = {
            "qid": qid,
            "label": ((entity.get("labels") or {}).get("en") or {}).get("value"),
            "description": ((entity.get("descriptions") or {}).get("en") or {}).get("value"),
            "url": entity_url(qid),
            "claims": entity.get("claims") or {},
        }
    return result


def _item_claims(entity: dict[str, Any], prop: str) -> list[str]:
    qids = []
    for claim in entity["claims"].get(prop, []):
        value = (claim.get("mainsnak") or {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and value.get("id"):
            qids.append(value["id"])
    return qids


def find_candidates(terms: list[str], fetch_json: FetchJson = _fetch_json) -> list[dict[str, Any]]:
    """Research-problem candidates for the terms, each with its model
    (or None if the problem has no P1513 link), best match first."""
    qids: list[str] = []
    for term in terms:
        for qid in search_items(term, RESEARCH_PROBLEM_CLASS, fetch_json):
            if qid not in qids:
                qids.append(qid)
    qids = qids[:MAX_CANDIDATES]
    problems = get_entities(qids, fetch_json)

    model_qids = {p: (_item_claims(problems[p], PROBLEM_TO_MODEL) or [None])[0] for p in problems}
    models = get_entities([m for m in model_qids.values() if m], fetch_json)

    candidates = []
    for qid in qids:
        if qid not in problems:
            continue
        problem = {k: v for k, v in problems[qid].items() if k != "claims"}
        model = models.get(model_qids.get(qid) or "")
        candidates.append({
            "problem": problem,
            "model": {k: v for k, v in model.items() if k != "claims"} if model else None,
        })
    return candidates
