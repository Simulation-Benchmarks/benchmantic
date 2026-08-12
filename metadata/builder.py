# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.builder

Orchestrates generation of the benchmark's RO-Crate JSON-LD ("benchmark.jsonld"):
the JSON-LD @context, the manifest (license/software/publisher/author
derivation), GraphBuilder (assembles the actual @graph), RO-Crate 1.1
conformance validation, and the CLI (parse_args/build) that ties it all
together. This is the module describe_benchmark.py calls directly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from ai.cache import (
    cache_path, load_cache, load_metric_cache, metric_cache_path,
    save_cache, save_metric_cache,
)
from ai import corrections as corrections_store
from ai.inference import DEFAULT_PROVIDER, PROVIDER_CONFIG, infer_metric_metadata, infer_parameter_metadata
import review
from metadata.metrics import (
    DEFAULT_METRIC_UNIT, KNOWN_METRIC_UNITS, build_metric_fields,
    discover_metrics_from_maincc,
)
from metadata.parameters import (
    DEFAULT_SCENARIO_SECTIONS, attach_cpp_hints, build_parameter_fields,
    default_scenario_candidates, discover_parameters, resolve_case_params,
)
from metadata.publication import extract_publication_citation
from metadata.repository import (
    discover_cases, extract_authors_file_hint, extract_authors_list,
    extract_benchmark_description, extract_class_label, extract_publisher_from_repo_url,
    extract_readme_dependencies, extract_readme_description, extract_readme_license,
    extract_readme_repo_url, extract_spdx_info, find_authors_file, find_executable_name,
    find_module_dir, find_readme,
)
from metadata.software import detect_software_label
from utils import read_text, slugify


DEFAULT_CONTEXT = {
    "local": "https://github.com/Simulation-Benchmarks/rotating-cylinders/",
    "@vocab": "http://w3id.org/nfdi4ing/metadata4ing#",
    "dcterms": "http://purl.org/dc/terms/",
    "dcat": "http://www.w3.org/ns/dcat#",
    # NOTE: must be "http://" (not "https://") -- the RO-Crate 1.1 profile's
    # SHACL shapes hard-code sh:hasValue schema_org:CreativeWork /
    # schema_org:Dataset using the http scheme, and RDF term equality is a
    # strict string match, so https://schema.org/... would silently fail
    # every RO-Crate root-entity check even though it's "the same" URI to a
    # human.
    "schema": "http://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "qudt": "http://qudt.org/schema/qudt/",
    "m4i": "http://w3id.org/nfdi4ing/metadata4ing#",
    "mathmod": "https://mardi4nfdi.de/mathmoddb#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    # NOTE: these two were previously missing, which silently broke any
    # JSON-LD consumer resolving CURIEs properly (e.g. semantic_benchmark's
    # BenchmarkLoader, which queries for the fully-expanded obo:BFO_0000051
    # "has part" IRI and finds nothing without this) -- "has part" and
    # "represents" below were pointing at undefined-prefix CURIEs that never
    # actually expanded to the IRIs those terms are supposed to mean.
    "obo": "http://purl.obolibrary.org/obo/",
    "sio": "http://semanticscience.org/resource/",
    "label": {"@id": "rdfs:label"},
    "Field": {"@id": "cr:Field"},
    "file object": {"@id": "cr:FileObject"},
    "method": {"@id": "m4i:Method"},
    "numerical variable": {"@id": "m4i:NumericalVariable"},
    # Distinct from "numerical variable" above: used for value-bearing
    # PARAMETER nodes (not metric/evaluates definitions). Deliberately NOT
    # m4i:NumericalVariable -- see add_parameter_variable()'s comment for
    # why. Mirrors semantic_benchmark's own NumericalParameter/TextParameter
    # dataclass names for clarity, though nothing currently requires this
    # exact IRI -- it just needs to not be m4i:NumericalVariable.
    "numerical parameter": {"@id": "m4i:NumericalParameter"},
    "text parameter": {"@id": "m4i:TextParameter"},
    "processing step": {"@id": "m4i:ProcessingStep"},
    "tool": {"@id": "m4i:Tool"},
    "has numerical value": {"@id": "m4i:hasNumericalValue"},
    "has string value": {"@id": "m4i:hasStringValue"},
    "has unit": {"@id": "m4i:hasUnit"},
    "has quantity kind": {"@id": "m4i:hasKindOfQuantity"},
    "has part": {"@id": "obo:BFO_0000051"},
    "has input": {"@id": "obo:RO_0002233"},
    "has output": {"@id": "obo:RO_0002234"},
    "has employed tool": {"@id": "m4i:hasEmployedTool"},
    "has configuration": {"@id": "m4i:usesConfiguration"},
    "has parameter set": {"@id": "m4i:hasParameterSet"},
    "evaluates": {"@id": "m4i:evaluates"},
    "investigates": {"@id": "m4i:investigates"},
    "uses": "mathmod:uses",
    "describedAsDocumentedBy": "mathmod:describedAsDocumentedBy",
    "extract": {"@id": "cr:extract"},
    "jsonPath": {"@id": "cr:jsonPath"},
    "source": {"@id": "cr:source"},
    "represents": {"@id": "sio:SIO_000210"},
    # RO-Crate root-entity / file-descriptor terms.
    #
    # "identifier" is ONLY used on ParameterSet nodes (the case id) in this
    # codebase -- mapped to m4i:identifier (not schema:identifier) because
    # that's what semantic_benchmark's BenchmarkLoader actually queries for
    # (M4I.identifier) when reading each configuration's identifier back
    # out; using schema:identifier here made every ParameterSet.identifier
    # resolve to None downstream, which in turn made
    # run_benchmark.py's create_parameter_files_from_benchmark() skip every
    # single configuration (it requires a truthy identifier).
    "identifier": {"@id": "m4i:identifier"},
    "dataType": {"@id": "cr:dataType"},
    "name": {"@id": "schema:name"},
    "description": {"@id": "schema:description"},
    "datePublished": {"@id": "schema:datePublished"},
    "license": {"@id": "schema:license"},
    "hasPart": {"@id": "schema:hasPart"},
    "about": {"@id": "schema:about"},
    "conformsTo": {"@id": "dcterms:conformsTo"},
    "author": {"@id": "schema:author"},
    "publisher": {"@id": "schema:publisher"},
    "codeRepository": {"@id": "schema:codeRepository"},
    "softwareRequirements": {"@id": "schema:softwareRequirements"},
}


#: Absolute fallbacks for fields we truly cannot derive from source (no
#: version-control tag, no "investigates" ontology mapping available from
#: code alone, etc). Everything else in the manifest is scraped from
#: problem.hh/main.cc by build_manifest() below.
MANIFEST_FALLBACKS: dict[str, Any] = {
    "version": "1.0.0",
    "label": "benchmark",
    "software_label": "simulation software",
    "publication_label": "Publication",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "license_label": "CC BY 4.0",
}


def build_manifest(
    problem_hh_text: str,
    main_cc_text: str,
    benchmark_description: str,
    readme_text: str = "",
    authors_text: str = "",
    authors_file_name: str | None = None,
) -> dict[str, Any]:
    """Derive the RO-Crate manifest (crate label, license, software, etc.)
    from problem.hh/main.cc instead of a hardcoded DEFAULT_MANIFEST or an
    external --manifest file. Falls back to MANIFEST_FALLBACKS for anything
    that genuinely isn't recoverable from source (e.g. a version number, or
    a formal "investigates" ontology/QID mapping).

    `readme_text` (optional) fills in a few fields the source files alone
    don't carry: a license fallback when no SPDX header is present, the
    repo's clone URL, the module's pinned dependency versions, and --
    when there's no better signal -- a best-effort publisher/author guess
    derived from the repo's hosting domain. `authors_text` (optional, the
    contents of an AUTHORS/CONTRIBUTORS file if one was found) is the
    highest-priority author source when present: real named contributors
    beat any guess. See extract_readme_license() / extract_readme_repo_url()
    / extract_readme_dependencies() / extract_publisher_from_repo_url() /
    extract_authors_list().
    """
    spdx = extract_spdx_info(problem_hh_text, main_cc_text)
    class_label = extract_class_label(problem_hh_text, main_cc_text)
    citation = extract_publication_citation(benchmark_description)
    readme_license_id = extract_readme_license(readme_text)
    repo_url = extract_readme_repo_url(readme_text)

    # Organization signal from the repo's hosting domain -- computed
    # unconditionally (not just as a fallback) because even when SPDX gives
    # us a real copyright-holder *name*, it never gives us a *URL*, and this
    # is the only source that can.
    org_guess = extract_publisher_from_repo_url(repo_url)

    #: An SPDX copyright header (e.g. "DuMux project contributors") is a
    #: real, asserted fact about who wrote/owns the code -- a legitimate
    #: source for publisher (and, absent an AUTHORS file, author too).
    #: Absent both, everything falls back to the repo-hosting-domain guess
    #: so the crate isn't left with no author/publisher at all.
    copyright_holder = spdx["copyright"]
    #: A copyright string naming a collective ("X project contributors",
    #: "X team", "X consortium", a company suffix, a university, ...) reads
    #: as an Organization; anything else defaults to Person, since a bare
    #: copyright line just as easily names an individual maintainer.
    looks_like_org = bool(re.search(
        r"\b(contributors?|project|team|consortium|university|institute|"
        r"gmbh|inc\.?|ltd\.?|corp\.?|foundation|group)\b",
        copyright_holder or "", re.IGNORECASE,
    ))

    # Author precedence: real named contributors from an AUTHORS/
    # CONTRIBUTORS file > the SPDX copyright holder > the repo-hosting-
    # domain guess > nothing.
    file_authors, authors_total = extract_authors_list(authors_text)
    if file_authors:
        authors, authors_source = file_authors, "authors_file"
    elif copyright_holder:
        authors, authors_source = [copyright_holder], "spdx"
    elif org_guess:
        authors, authors_source = [org_guess["name"]], "repo_guess"
    else:
        authors, authors_source = [], None

    publisher_name = copyright_holder or (org_guess["name"] if org_guess else None)
    publisher_name_is_guess = not copyright_holder and org_guess is not None

    manifest: dict[str, Any] = {
        "label": class_label or MANIFEST_FALLBACKS["label"],
        "version": MANIFEST_FALLBACKS["version"],
        # No source-derivable mapping to a formal ontology/QID for the
        # physical phenomenon under study -- GraphBuilder falls back to a
        # local id built from investigates_label when this is None.
        "investigates_qid": None,
        "investigates_label": class_label or MANIFEST_FALLBACKS["label"],
        "software_label": detect_software_label(problem_hh_text, main_cc_text) or MANIFEST_FALLBACKS["software_label"],
        "publication_label": citation or MANIFEST_FALLBACKS["publication_label"],
        "root_description": benchmark_description or None,
        "date_published": None,
        "authors": authors,
        "authors_source": authors_source,  # "authors_file" | "spdx" | "repo_guess" | None
        "authors_omitted": max(0, authors_total - len(file_authors)),
        "authors_file_name": authors_file_name,
        "author_type": "schema:Person" if (copyright_holder and not looks_like_org) else "schema:Organization",
        # The URL always comes from the repo-hosting-domain guess (SPDX and
        # AUTHORS files never carry one) -- attached to the single-author
        # fallback cases even when the *name* itself came from a real SPDX
        # header, since a URL is better than none.
        "author_url": org_guess["url"] if org_guess else None,
        "publisher_name": publisher_name,
        "publisher_url": org_guess["url"] if org_guess else None,
        "publisher_name_is_guess": publisher_name_is_guess,
        # Repo clone URL and pinned dependency versions -- only ever
        # recoverable from a README (source files don't carry this), so
        # these are None whenever no README was found or it didn't match
        # the expected patterns.
        "repo_url": repo_url,
        "dependencies": extract_readme_dependencies(readme_text),
    }

    # License precedence: an SPDX header in the actual source is the most
    # trustworthy signal (it's machine-checked in the repo itself); a REUSE-
    # style license link in the README is the next best thing; only fall
    # back to the generic default if neither is present.
    if spdx["license_id"]:
        manifest["license_url"] = f"https://spdx.org/licenses/{spdx['license_id']}.html"
        manifest["license_label"] = spdx["license_id"]
    elif readme_license_id:
        manifest["license_url"] = f"https://spdx.org/licenses/{readme_license_id}.html"
        manifest["license_label"] = readme_license_id
    else:
        manifest["license_url"] = MANIFEST_FALLBACKS["license_url"]
        manifest["license_label"] = MANIFEST_FALLBACKS["license_label"]

    return manifest



def _format_dependency(dep: dict[str, str]) -> str:
    """Render one extract_readme_dependencies() row as a short human-
    readable string, e.g. 'dune-istl@21c67275b17e (origin/releases/2.10,
    2025-02-03 09:13:05 +0000)'.
    """
    module = dep.get("module") or "?"
    commit = dep.get("commit") or ""
    text = f"{module}@{commit[:12]}" if commit else module
    extra = ", ".join(v for v in (dep.get("branch"), dep.get("date")) if v)
    return f"{text} ({extra})" if extra else text



class GraphBuilder:
    def __init__(
        self,
        manifest: dict[str, Any],
        parameter_fields: dict[str, Any],
        metric_fields: dict[str, Any] | None = None,
        benchmark_description: str = "",
    ):
        self.manifest = manifest
        self.parameter_fields = parameter_fields
        #: LLM-inferred (or cached) metric metadata, keyed by raw metric key
        #: as it appears in main.cc / the summary JSON. See build_metric_fields().
        self.metric_fields = metric_fields or {}
        #: Doc-comment (e.g. Doxygen \brief + citation) scraped from
        #: problem.hh/main.cc -- see extract_benchmark_description(). Used as
        #: a human-readable description on the top-level benchmark node.
        self.benchmark_description = benchmark_description
        #: Graph id for the benchmark node -- derived from the manifest
        #: label (itself scraped from problem.hh's class name) rather than
        #: hardcoded, so this stays correct for any benchmark, not just
        #: rotating-cylinders.
        self.benchmark_id = f"local:bm-{slugify(manifest.get('label', 'benchmark'))}"
        self.graph: list[dict[str, Any]] = []
        self._param_value_nodes: dict[tuple[str, Any], str] = {}
        self._extract_nodes: set[str] = set()
        self._metric_fields_built = False

    def _ensure_extract_node(self, key: str) -> str:
        extract_id = f"local:extract_{key}"
        if extract_id not in self._extract_nodes:
            self.graph.append({
                "@id": extract_id,
                "@type": "cr:DataSource",
                "jsonPath": f"/{key}",
            })
            self._extract_nodes.add(extract_id)
        return extract_id

    def add_parameter_variable(self, semantic_name: str, value: Any, spec: dict[str, Any]) -> str:
        # @id/label are tied to the actual params.input identity (section +
        # key) rather than the LLM-invented semantic_name, so they stay
        # stable and traceable back to the source file regardless of what
        # the model chose to call the parameter. semantic_name and the
        # LLM's explanation become a human-readable description instead.
        section, ini_key = spec["ini"]
        raw_label = f"{section}.{ini_key}"
        json_key = spec.get("json_key", ini_key)
        # Lists aren't hashable, so use a tuple for the dedup key and a
        # joined string for the id slug when this is a full_value parameter
        # (see resolve_case_params()/--full-value-params).
        dedup_value = tuple(value) if isinstance(value, list) else value
        dedup_key = (raw_label, dedup_value)
        if dedup_key in self._param_value_nodes:
            return self._param_value_nodes[dedup_key]

        slug_value = "_".join(str(v) for v in value) if isinstance(value, list) else value
        suffix = f"{slugify(section)}_{slugify(ini_key)}_{slugify(slug_value)}"
        var_id = f"local:variable_{suffix}"
        field_id = f"local:field_{suffix}"
        source_id = f"local:source_{suffix}"
        extract_id = self._ensure_extract_node(json_key)

        description = spec.get("description") or ""
        if semantic_name and semantic_name != raw_label:
            description = f"{description} (inferred semantic name: {semantic_name})".strip()

        # NOTE: deliberately NOT typed "numerical variable" (m4i:NumericalVariable)
        # here, unlike metric nodes in ensure_metric_fields(). Downstream,
        # semantic_benchmark.BenchmarkLoader.build_parameter_entry() routes
        # any node typed m4i:NumericalVariable into its value-less
        # NumericalVariable dataclass (no "has numerical value" is ever
        # read for that type -- it's meant for metric *definitions*, which
        # have no value until after a run). An untyped node with
        # hasNumericalValue/hasStringValue instead becomes a
        # NumericalParameter/TextParameter, which DOES carry the value.
        # Typing parameter nodes as NumericalVariable would silently make
        # every parameter value resolve to None downstream.
        var_node = {
            "@id": var_id,
            "label": raw_label,
            "dcterms:description": description,
            "has unit": {"@id": spec["unit"]},
        }
        if isinstance(value, list):
            # A full_value (multi-token) parameter -- stored as a single
            # space-joined string via "has string value" rather than
            # multiple "has numerical value" triples, because RDF has no
            # concept of list order and a single graph.value() lookup
            # (which is what consumers use) only ever returns ONE of
            # several same-predicate triples. Consumers that need the
            # individual numbers back can .split() this string.
            var_node["@type"] = "text parameter"
            var_node["has string value"] = " ".join(str(v) for v in value)
        elif spec.get("datatype") == "schema:String":
            var_node["@type"] = "text parameter"
            var_node["has string value"] = value
        else:
            var_node["@type"] = "numerical parameter"
            var_node["has numerical value"] = value
        if spec.get("quantityKind"):
            var_node["has quantity kind"] = {"@id": spec["quantityKind"]}

        self.graph += [
            var_node,
            {
                "@id": field_id, "@type": "Field",
                "dataType": {"@id": spec.get("datatype", "schema:Float")},
                "represents": {"@id": var_id},
                "source": {"@id": source_id},
            },
            {
                "@id": source_id, "@type": "cr:DataSource",
                "extract": {"@id": extract_id},
                "file object": {"@id": "local:parameter_file_object"},
            },
        ]
        self._param_value_nodes[dedup_key] = var_id
        return var_id

    def ensure_metric_fields(self, metric_keys: list[str]) -> list[str]:
        if self._metric_fields_built:
            raise RuntimeError("ensure_metric_fields() called twice")
        metric_ids = []
        for key in metric_keys:
            # Precedence: manual override (KNOWN_METRIC_UNITS) > LLM-inferred
            # (self.metric_fields, built via build_metric_fields()) > unitless
            # fallback, in case a key somehow wasn't inferred.
            spec = KNOWN_METRIC_UNITS.get(key) or self.metric_fields.get(key)
            if spec is None:
                print(f"warning: '{key}' (from main.cc) has no known or "
                      f"inferred unit annotation; defaulting to unitless.", file=sys.stderr)
                spec = DEFAULT_METRIC_UNIT
            # @id/label stay tied to the raw metric key exactly as it appears
            # in main.cc / the summary JSON file; the LLM's semantic_name and
            # explanation become a description instead.
            semantic_name = spec.get("semantic_name", key)
            description = spec.get("description") or ""
            if semantic_name and semantic_name != key:
                description = f"{description} (inferred semantic name: {semantic_name})".strip()
            slug = slugify(key)
            metric_id = f"local:metric_{slug}"
            field_id = f"local:field_{slug}"
            source_id = f"local:source_{slug}"
            extract_id = self._ensure_extract_node(key)
            self.graph += [
                {
                    "@id": metric_id, "@type": "numerical variable", "label": key,
                    "dcterms:description": description,
                    "has unit": {"@id": spec["unit"]},
                    **({"has quantity kind": {"@id": spec["quantityKind"]}}
                       if spec.get("quantityKind") else {}),
                },
                {
                    "@id": field_id, "@type": "Field",
                    "dataType": {"@id": spec.get("datatype", "schema:Double")},
                    "represents": {"@id": metric_id},
                    "source": {"@id": source_id},
                },
                {
                    "@id": source_id, "@type": "cr:DataSource",
                    "extract": {"@id": extract_id},
                    "file object": {"@id": "local:summary_file_object"},
                },
            ]
            metric_ids.append(metric_id)
        self._metric_fields_built = True
        return metric_ids

    def add_configuration(self, case_id: str, label: str, param_values: dict[str, Any]) -> str:
        var_ids = []
        for key, value in param_values.items():
            spec = self.parameter_fields[key]
            var_ids.append(self.add_parameter_variable(spec.get("semantic_name", key), value, spec))
        config_id = f"local:configuration_{case_id}"
        self.graph.append({
            "@id": config_id,
            "@type": "m4i:ParameterSet",
            "label": label,
            "identifier": case_id,
            "has part": [{"@id": v} for v in var_ids],
        })
        return config_id

    def add_benchmark_node(self, config_ids: list[str], metric_ids: list[str]) -> None:
        m = self.manifest
        # No source-derivable mapping to a formal ontology/QID for the
        # physical phenomenon under study, so fall back to a local id built
        # from investigates_label (see build_manifest()).
        investigates_id = m.get("investigates_qid") or f"local:investigates_{slugify(m['investigates_label'])}"
        self.graph.insert(0, {
            "@id": self.benchmark_id,
            "@type": "m4i:Benchmark",
            "label": m["label"],
            **({"dcterms:description": self.benchmark_description} if self.benchmark_description else {}),
            "investigates": {"@id": investigates_id},
            "uses": {"@id": investigates_id},
            "evaluates": [{"@id": i} for i in metric_ids],
            "has parameter set": [{"@id": c} for c in config_ids],
            "describedAsDocumentedBy": {"@id": "local:publication"},
            "schema:version": m["version"],
        })
        self.graph.append({
            "@id": investigates_id,
            "@type": "mathmod:ResearchProblem",
            "label": m["investigates_label"],
        })
        self.graph.append({
            "@id": "local:publication",
            "@type": "mathmod:Publication",
            "label": m["publication_label"],
        })
        self.graph.append({
            "@id": "local:software", "@type": "tool", "label": m["software_label"],
            **({
                "schema:softwareRequirements": [
                    _format_dependency(d) for d in m["dependencies"]
                ]
            } if m.get("dependencies") else {}),
        })
        self.graph.append({
            "@id": "local:parameter_file_object", "@type": "cr:FileObject",
            "label": "parameter.json",
        })
        self.graph.append({
            "@id": "local:summary_file_object", "@type": "cr:FileObject",
            "label": "solution_metrics.json",
        })

        # One m4i:ProcessingStep per case, linking each configuration to the
        # "run the simulation" step that consumes it. Required by
        # create_rocrate.py's aggregate-crate builder -- it locates
        # processing steps by RDF type (not via a link from the Benchmark
        # node) and raises ValueError("Benchmark has no processing steps.")
        # if there are none. usesConfiguration ("has configuration") is
        # what actually ties a step back to its m4i:ParameterSet; input/
        # output/employed-tool reuse the file-object and software nodes
        # already in the graph.
        for config_id in config_ids:
            case_id = config_id.removeprefix("local:configuration_")
            step_id = f"local:processing_step_{case_id}"
            self.graph.append({
                "@id": step_id,
                "@type": "processing step",
                "label": f"Run simulation ({case_id})",
                "has configuration": {"@id": config_id},
                "has input": {"@id": "local:parameter_file_object"},
                "has output": {"@id": "local:summary_file_object"},
                "has employed tool": {"@id": "local:software"},
            })

    def add_rocrate_root(self) -> None:
        """Add the RO-Crate 1.1 Metadata File Descriptor and Root Data
        Entity so this graph is a *bona fide* RO-Crate, not just a bag of
        metadata4ing/Croissant nodes -- see
        https://www.researchobject.org/ro-crate/1.1/root-data-entity.html.
        Call this last, after add_benchmark_node(), so it can reference the
        benchmark and file-object entities that already exist in the graph.
        """
        m = self.manifest
        root_id = "./"
        license_id = m.get("license_url") or "local:license"

        root_entity: dict[str, Any] = {
            "@id": root_id,
            "@type": "schema:Dataset",
            "name": m["label"],
            "description": m.get("root_description") or self.benchmark_description or m["label"],
            "datePublished": m.get("date_published") or _dt.date.today().isoformat(),
            "license": {"@id": license_id},
            "hasPart": [
                {"@id": self.benchmark_id},
                {"@id": "local:parameter_file_object"},
                {"@id": "local:summary_file_object"},
            ],
        }
        if m.get("repo_url"):
            root_entity["codeRepository"] = m["repo_url"]

        def _guess_note(name_is_guess: bool, has_url: bool) -> str | None:
            if name_is_guess:
                return (
                    "Automatically inferred from the repository's hosting domain "
                    "(no SPDX copyright header was found in source) -- please verify."
                )
            if has_url:
                return "URL automatically inferred from the repository's hosting domain -- please verify."
            return None

        publisher_id = None
        if m.get("publisher_name"):
            publisher_id = f"local:publisher_{slugify(m['publisher_name'])}"
            root_entity["publisher"] = {"@id": publisher_id}
            publisher_node = {
                "@id": publisher_id,
                "@type": "schema:Organization",
                "name": m["publisher_name"],
            }
            if m.get("publisher_url"):
                publisher_node["schema:url"] = m["publisher_url"]
            note = _guess_note(m.get("publisher_name_is_guess", False), bool(m.get("publisher_url")))
            if note:
                publisher_node["schema:disambiguatingDescription"] = note
            self.graph.append(publisher_node)

        authors = m.get("authors") or []
        if authors:
            source = m.get("authors_source")
            node_type = "schema:Person" if source == "authors_file" else m.get("author_type", "schema:Organization")
            author_refs = []
            for name in authors:
                author_id = f"local:author_{slugify(name)}"
                author_node = {"@id": author_id, "@type": node_type, "name": name}
                if source != "authors_file" and m.get("author_url"):
                    author_node["schema:url"] = m["author_url"]
                # Named individuals (from an AUTHORS/CONTRIBUTORS file) are
                # presumed affiliated with the crate's publisher org, when
                # we have one -- that's the best signal available; there's
                # no per-person institution info in a plain AUTHORS file.
                if node_type == "schema:Person" and publisher_id:
                    author_node["schema:affiliation"] = {"@id": publisher_id}
                if source == "repo_guess":
                    author_node["schema:disambiguatingDescription"] = (
                        "Automatically inferred from the repository's hosting domain "
                        "(no SPDX copyright header or AUTHORS file was found) -- please verify."
                    )
                self.graph.append(author_node)
                author_refs.append({"@id": author_id})

            # Long AUTHORS/CONTRIBUTORS files are truncated (see
            # extract_authors_list()'s `limit`) -- represent the remainder
            # as one extra node rather than silently dropping them.
            if source == "authors_file" and m.get("authors_omitted"):
                more_id = "local:author_additional_contributors"
                self.graph.append({
                    "@id": more_id,
                    "@type": "schema:Organization",
                    "name": f"{m['authors_omitted']} additional contributor(s)",
                    "schema:disambiguatingDescription":
                        f"See {m.get('authors_file_name') or 'the authors file'} for the full list.",
                })
                author_refs.append({"@id": more_id})

            root_entity["author"] = author_refs if len(author_refs) > 1 else author_refs[0]

        self.graph.insert(0, root_entity)
        self.graph.insert(0, {
            "@id": "ro-crate-metadata.json",
            "@type": "schema:CreativeWork",
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
            "about": {"@id": root_id},
        })
        self.graph.append({
            "@id": license_id,
            "@type": "schema:CreativeWork",
            "name": m.get("license_label") or license_id,
        })


# =============================================================================
# 4. Case discovery & Resolution
# =============================================================================



def validate_rocrate(doc: dict[str, Any], severity: str = "REQUIRED") -> bool:
    """Validate `doc` (the full {"@context", "@graph"} document) against the
    RO-Crate 1.1 profile. Prints a completeness percentage plus a pass/fail
    summary (at the requested `severity` threshold) and any issues found.
    Returns True if validation passed at that threshold (or the validator
    isn't installed -- this check is informational and never aborts
    metadata generation).

    Note on "completeness": this is the fraction of the *public* RO-Crate
    1.1 profile's checks (REQUIRED+RECOMMENDED+OPTIONAL) that pass -- the
    same idea as the completeness bar shown for Research Objects registered
    in ROHub (https://www.rohub.org), but not the same number. ROHub's own
    score comes from its proprietary, server-side MINIM-based checklist
    service, computed only once a Research Object is actually
    uploaded/registered there, and isn't reproducible offline.
    """
    try:
        from rocrate_validator import services, models
        from rocrate_validator.models.settings import ValidationSettings
    except ImportError:
        print(
            "\nSkipping RO-Crate validation: the 'rocrate_validator' package "
            "isn't installed. Run `pip install roc-validator` to enable it "
            "(https://github.com/crs4/rocrate-validator).",
            file=sys.stderr,
        )
        return True

    print("\nValidating RO-Crate against profile 'ro-crate-1.1'...")

    # Always evaluate the full REQUIRED+RECOMMENDED+OPTIONAL check set so we
    # can report a completeness percentage regardless of which severity
    # threshold the caller cares about for pass/fail.
    settings = ValidationSettings(
        profile_identifier="ro-crate-1.1",
        metadata_only=True,
        metadata_dict=doc,
        requirement_severity="OPTIONAL",
    )
    result = services.validate_metadata_as_dict(doc, settings)

    stats = result.statistics.to_dict()
    checks = stats["checks"]
    completeness = checks["passed"]["percentage"]
    print(
        f"Completeness: {completeness:.1f}% of RO-Crate 1.1 checks passed "
        f"({checks['passed']['count']}/{checks['count']} checks, REQUIRED+RECOMMENDED+OPTIONAL)."
    )

    threshold = getattr(models.Severity, severity)
    all_issues = result.get_issues()
    passed_at_threshold = result.passed(min_severity=threshold)

    if passed_at_threshold:
        print(f"RO-Crate validation PASSED at severity >= {severity}.")
    else:
        print(f"RO-Crate validation FAILED at severity >= {severity}.")

    if all_issues:
        print(f"\nFailing checks ({len(all_issues)} of {checks['count']} total -- this is the gap behind the "
              f"{100 - completeness:.1f}% incomplete):")
        for issue in all_issues:
            blocking = " [BLOCKS PASS]" if issue.severity >= threshold else ""
            print(f"  [{issue.severity.name}]{blocking} {issue.check.identifier}: {issue.message}", file=sys.stderr)
            if issue.violatingEntity:
                print(f"      entity: {issue.violatingEntity}"
                      + (f"  property: {issue.violatingProperty}" if issue.violatingProperty else ""),
                      file=sys.stderr)
    else:
        print("No issues at any severity -- crate is 100% complete against this profile.")

    return passed_at_threshold




# =============================================================================
# 5. Execution Orchestration
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module_dir", type=Path,
                     help="Path to the benchmark module folder (containing main.cc, problem.hh, and "
                          "params.input file(s)), OR a higher-level repo/checkout directory -- the script "
                          "will search recursively for the main.cc+problem.hh pair.")
    ap.add_argument("--main-cc", type=Path, default=None, dest="main_cc",
                     help="Explicit path to main.cc, only needed if more than one main.cc+problem.hh "
                          "pair is found under module_dir.")
    ap.add_argument("--output", type=Path, default=Path("metadata.jsonld"))
    ap.add_argument("--scenario-params", type=str, default=None,
                     help="Comma-separated list of raw parameter keys (e.g., 'Cells0,Cells1') that are scenario-specific. "
                          "If omitted, you will be prompted interactively.")
    ap.add_argument("--full-value-params", type=str, default=None,
                     help="Comma-separated list of raw parameter keys (e.g., 'Radial0') whose params.input value has "
                          "MULTIPLE whitespace-separated tokens (e.g. 'Radial0 = 1.0 1.5 2.0') that should ALL be "
                          "captured as a list, instead of collapsing to a single value via the usual LLM-picked "
                          "token index. Use this for parameters like a multi-value radius/coordinate list where "
                          "every token matters, not just one.")
    ap.add_argument("--provider", type=str, choices=sorted(PROVIDER_CONFIG), default=DEFAULT_PROVIDER,
                     help=f"LLM provider used to infer parameter metadata (default: {DEFAULT_PROVIDER}). "
                          "Groq is free and requires GROQ_API_KEY; OpenAI requires OPENAI_API_KEY and billing.")
    ap.add_argument("--model", type=str, default=None,
                     help="Model name to use for the chosen --provider. Defaults to "
                          f"'{PROVIDER_CONFIG['groq']['default_model']}' for groq or "
                          f"'{PROVIDER_CONFIG['openai']['default_model']}' for openai.")
    ap.add_argument("--fallback-on-error", action="store_true",
                     help="If the LLM call fails (e.g. quota exceeded, network error), fall back to "
                          "local placeholder metadata (unitless, confidence 0.0) instead of exiting. "
                          "These placeholders are NOT written to the cache, so a future run will retry the API.")
    ap.add_argument("--verbose", action="store_true",
                     help="Print the request details and raw LLM response for each API call "
                          "(endpoint, model, token usage, timing, and the exact response text).")
    ap.add_argument("--clear-cache", action="store_true",
                     help="Delete any existing parameter/metric metadata cache for this module "
                          "before running, forcing fresh LLM inference for every selected item. "
                          "Without this flag, stale cache entries that no longer correspond to a "
                          "real params.input key or main.cc metric are pruned automatically, but "
                          "still-valid cached entries are reused as normal.")
    ap.add_argument("--skip-review", action="store_true",
                     help="Skip the interactive review/edit step after AI inference and accept the "
                          "inferred parameter/metric metadata as-is. Required for non-interactive/CI "
                          "runs (this step needs a real terminal for input()), same as --scenario-params.")
    ap.add_argument("--review-confidence-threshold", type=float, default=review.DEFAULT_CONFIDENCE_THRESHOLD,
                     help=f"During review, flag any parameter/metric with confidence below this value and "
                          f"refuse a plain 'yes' at the final-confirmation prompt until it's either fixed "
                          f"or explicitly accepted by typing 'accept' (default: {review.DEFAULT_CONFIDENCE_THRESHOLD}). "
                          "Has no effect with --skip-review.")
    ap.add_argument("--skip-validation", action="store_true",
                     help="Skip the automatic RO-Crate 1.1 conformance check that normally runs "
                          "after writing the output (via the 'rocrate_validator' package -- "
                          "`pip install roc-validator`). Use this if the package isn't installed "
                          "or you don't have network access for it to fetch the profile.")
    ap.add_argument("--validate-severity", type=str, default="REQUIRED",
                     choices=["OPTIONAL", "RECOMMENDED", "REQUIRED"],
                     help="Minimum issue severity to check/report for the RO-Crate conformance "
                          "check (default: REQUIRED, i.e. only what the RO-Crate spec mandates -- "
                          "pass RECOMMENDED or OPTIONAL for stricter best-practice checks).")
    return ap


def build(args: argparse.Namespace) -> None:
    if not args.module_dir.is_dir():
        sys.exit(f"Error: {args.module_dir} is not a directory")

    # 1. Path Resolution -- module_dir may be the exact benchmark folder or a
    # higher-level repo checkout; resolve down to the directory that
    # actually holds main.cc + problem.hh, and locate a README nearby.
    # Everything below (cache files, params.input discovery, case discovery)
    # is scoped to this resolved module directory, NOT the (possibly much
    # larger) repo root, so pointing this at a whole repo checkout doesn't
    # accidentally pull in cases/params from unrelated benchmarks elsewhere
    # in the same repo.
    repo_root = args.module_dir
    module_dir = find_module_dir(repo_root, args.main_cc)
    if module_dir != repo_root:
        print(f"Resolved benchmark module: {module_dir}")

    main_cc_path = module_dir / "main.cc"
    problem_hh_path = module_dir / "problem.hh"

    readme_path = find_readme(module_dir, repo_root)
    if readme_path:
        print(f"Found README: {readme_path}")

    authors_hint = extract_authors_file_hint(read_text(problem_hh_path), read_text(main_cc_path))
    authors_path = find_authors_file(module_dir, repo_root, hint=authors_hint)
    if authors_path:
        print(f"Found authors file: {authors_path}")

    executable_name, cmakelists_path = find_executable_name(module_dir, repo_root)
    if executable_name:
        print(f"Found executable target: {executable_name} (from {cmakelists_path})")
    else:
        print(
            "warning: could not find a CMakeLists.txt anywhere under the repo declaring an "
            "executable/test target for main.cc -- generate_snakefile.py will need --executable "
            "passed explicitly.",
            file=sys.stderr,
        )
    # module_dir's path relative to the repo root -- e.g.
    # "test/freeflow/navierstokes/rotatingcylinders" -- not an RO-Crate
    # field, but useful build-workflow info; written to the sidecar file
    # below (see BUILD_HINTS_SUFFIX) rather than into metadata.jsonld itself.
    module_relative_path = str(module_dir.relative_to(repo_root)) if module_dir != repo_root else None

    if args.clear_cache:
        for p in (cache_path(module_dir), metric_cache_path(module_dir)):
            if p.exists():
                p.unlink()
                print(f"Cleared cache: {p}")

    params_input_path = module_dir / "params.input"

    if not params_input_path.exists():
        discovered = sorted(module_dir.rglob("params.input"))
        if discovered:
            params_input_path = discovered[0]
        else:
            sys.exit(f"Error: Could not find a template params.input under {module_dir}")

    # 2. Discover Raw Parameters from INI file first (Before LLM)
    raw_candidates = discover_parameters(params_input_path)
    if not raw_candidates:
        sys.exit("Error: No raw parameters found in the template params.input file.")

    # 2b. Enrich candidates with getParam<Type>("Section.Key") call-site hints
    #     scraped from main.cc/problem.hh -- e.g. "Component.LiquidDensity" ->
    #     assigned to `density_` (C++ type `Scalar`). This gives the LLM
    #     code-grounded evidence for picking the right SI unit, rather than
    #     just guessing from the INI key name.
    attach_cpp_hints(raw_candidates, read_text(main_cc_path), read_text(problem_hh_path))

    # Doc-comment (e.g. Doxygen \brief + literature citation) scraped from
    # problem.hh/main.cc -- passed to the LLM as high-level scenario context
    # for both parameter and metric inference, and recorded on the
    # benchmark graph node itself.
    benchmark_description = extract_benchmark_description(
        read_text(problem_hh_path), read_text(main_cc_path)
    )
    if not benchmark_description and readme_path:
        benchmark_description = extract_readme_description(read_text(readme_path))

    # 3. Filter raw candidates to determine which are Scenario-Specific
    selected_candidates = []
    raw_map = {c.key.lower(): c for c in raw_candidates}

    if args.scenario_params:
        provided_keys = [k.strip().lower() for k in args.scenario_params.split(",") if k.strip()]
        for pk in provided_keys:
            if pk in raw_map:
                selected_candidates.append(raw_map[pk])
            else:
                print(f"Warning: Param '{pk}' provided in --scenario-params was not found in params.input. Skipping.", file=sys.stderr)
        if not selected_candidates:
            # --scenario-params was explicitly given, so we never fall back
            # to the interactive prompt (that would silently ignore the
            # user's non-interactive intent, e.g. in a scripted/CI run).
            # Fail loudly instead.
            sys.exit(
                "Error: None of the parameters specified in --scenario-params "
                f"('{args.scenario_params}') were found in params.input. "
                f"Available keys: {', '.join(sorted(c.key for c in raw_candidates))}"
            )

    # Only enter interactive selection when --scenario-params was not given
    # at all; if it was given, `selected_candidates` is already final
    # (non-empty, or we've already exited above).
    if not args.scenario_params and not selected_candidates:
        default_candidates = default_scenario_candidates(raw_candidates)
        default_indices = {
            i for i, c in enumerate(raw_candidates) if c in default_candidates
        }
        sections_label = ", ".join(sorted(DEFAULT_SCENARIO_SECTIONS))
        checked_indices = set(default_indices)

        while True:
            print("\n=== Parameter Selection ===")
            print(f"Parameters under [{sections_label}] are pre-selected by default (marked [x]).\n")

            # Print tabular structure (max 2 columns) with checkbox markers
            cols = 2
            longest_formatted_len = max(len(f"[x] {i:2d} {c.key}") for i, c in enumerate(raw_candidates))
            col_width = longest_formatted_len + 4  # padding margin

            for r_idx in range(0, len(raw_candidates), cols):
                chunk = raw_candidates[r_idx:r_idx + cols]
                row_str = "".join(
                    f"[{'x' if (r_idx + idx) in checked_indices else ' '}] {r_idx + idx:2d} {cand.key}".ljust(col_width)
                    for idx, cand in enumerate(chunk)
                )
                print("  " + row_str)

            print("\nInstructions:")
            print("  - Type index numbers to toggle them on/off (e.g., 0, 3, 5)")
            print("  - Type 'all' to select everything, or 'none' to clear the selection")
            print("  - Repeat as many times as needed -- each entry toggles from where you left off")
            print("  - Finally, press Enter with no input to confirm the selection and proceed")

            try:
                user_input = input("\nToggle selection (or Enter to confirm): ").strip().lower()

                if not user_input:
                    if not checked_indices:
                        print("No parameters selected. Please select at least one.", file=sys.stderr)
                        continue
                    selected_candidates = [c for i, c in enumerate(raw_candidates) if i in checked_indices]
                    break

                if user_input == "all":
                    checked_indices = set(range(len(raw_candidates)))
                    continue
                if user_input == "none":
                    checked_indices = set()
                    continue

                toggled = [int(tok.strip()) for tok in user_input.split(",") if tok.strip().lstrip("-").isdigit()]
                valid_toggled = [i for i in toggled if 0 <= i < len(raw_candidates)]
                if not valid_toggled:
                    print("No valid indices recognized. Please try again.", file=sys.stderr)
                    continue
                for i in valid_toggled:
                    checked_indices.symmetric_difference_update({i})

            except (EOFError, KeyboardInterrupt):
                print(
                    f"\nInput cancelled. Falling back to the default: parameters under [{sections_label}].",
                    file=sys.stderr,
                )
                selected_candidates = default_candidates
                break

    print(f"\nFinal Selection Confirmed.")
    print(f"Scenario-Specific: {', '.join(c.key for c in selected_candidates)}")
    print(f"Tool-specific:   {', '.join(c.key for c in raw_candidates if c not in selected_candidates) or 'None'}\n")

    # 4. Cache Management & AI Inference (ONLY for the selected parameters)
    metadata_cache = load_cache(module_dir) or []

    # Prune any cached entries that no longer correspond to a real [section]
    # key in the current params.input -- e.g. leftovers from a prior buggy
    # run where the LLM hallucinated extra parameters it merely noticed in
    # the source code. Left in place, those can silently resurface and crash
    # resolve_case_params() later, exactly as happened before this check
    # existed. Genuinely valid cached entries are kept and NOT re-queried.
    valid_param_keys = {(c.section, c.key) for c in raw_candidates}
    pruned_metadata_cache = [item for item in metadata_cache if tuple(item["ini"]) in valid_param_keys]
    if len(pruned_metadata_cache) != len(metadata_cache):
        dropped = len(metadata_cache) - len(pruned_metadata_cache)
        print(
            f"Pruned {dropped} stale parameter cache entr{'y' if dropped == 1 else 'ies'} "
            "no longer present in params.input.",
            file=sys.stderr,
        )
        save_cache(module_dir, pruned_metadata_cache)
    metadata_cache = pruned_metadata_cache

    cache_lookup = {(item["ini"][0], item["ini"][1]): item for item in metadata_cache}
    
    final_metadata = []
    missing_candidates = []
    
    for candidate in selected_candidates:
        lookup_key = (candidate.section, candidate.key)
        if lookup_key in cache_lookup:
            final_metadata.append(cache_lookup[lookup_key])
        else:
            missing_candidates.append(candidate)
            
    if missing_candidates:
        resolved_model = args.model or PROVIDER_CONFIG[args.provider]["default_model"]
        print(f"\nQuerying {args.provider} ({resolved_model}) for {len(missing_candidates)} parameter(s) not found in cache...")
        # Prior human corrections to similarly named/typed parameters (from
        # ANY module reviewed with this checkout before) -- included as
        # guidance in the prompt so the model doesn't repeat a mistake a
        # human already fixed once. See ai.corrections / review.py.
        known_corrections = corrections_store.relevant_corrections_for(
            [c.key for c in missing_candidates], "parameter",
        )
        try:
            new_inferred = infer_parameter_metadata(
                candidates=missing_candidates,
                main_cc=read_text(main_cc_path),
                problem_hh=read_text(problem_hh_path),
                benchmark_description=benchmark_description,
                provider=args.provider,
                model=args.model,
                verbose=args.verbose,
                known_corrections=known_corrections,
            )
        except (RuntimeError, ValueError) as exc:
            if not args.fallback_on_error:
                sys.exit(
                    f"Error: {args.provider} parameter inference failed -- {exc}\n"
                    "Pass --fallback-on-error to use local placeholder metadata instead of exiting."
                )

            print(
                f"Warning: {args.provider} inference failed ({exc}). "
                f"Using local placeholder metadata for {len(missing_candidates)} parameter(s) instead.",
                file=sys.stderr,
            )
            new_inferred = [
                {
                    "semantic_name": candidate.key,
                    "ini": [candidate.section, candidate.key],
                    "index": 0,
                    "datatype": "schema:Float",
                    "unit": "unit:UNITLESS",
                    "quantityKind": None,
                    "confidence": 0.0,
                    "explanation": "Placeholder generated because the OpenAI API call failed.",
                }
                for candidate in missing_candidates
            ]
            final_metadata.extend(new_inferred)
            # Deliberately NOT written to the cache -- placeholders are low-confidence
            # stand-ins, so a future run should retry the API rather than reuse them.
        else:
            final_metadata.extend(new_inferred)
            # NOTE: not saved to the on-disk cache here -- final_metadata may
            # still be edited during the review step below, and the cache
            # should hold the FINAL (possibly human-corrected) values, not
            # the raw AI output. See the save_cache() call after review.
    else:
        print("Using cached parameter metadata (loaded from local file, 0 API queries triggered).")

    if args.full_value_params:
        full_value_keys = {k.strip().lower() for k in args.full_value_params.split(",") if k.strip()}
        matched = 0
        for item in final_metadata:
            if item["ini"][1].lower() in full_value_keys:
                item["full_value"] = True
                matched += 1
        unmatched = full_value_keys - {item["ini"][1].lower() for item in final_metadata if item.get("full_value")}
        if unmatched:
            print(
                f"Warning: --full-value-params key(s) not found among selected parameters: "
                f"{', '.join(sorted(unmatched))}",
                file=sys.stderr,
            )
        print(f"Capturing full multi-token value for {matched} parameter(s): {args.full_value_params}")

    # 4c. Interactive review -- show the table, let the reviewer edit any
    # field, loop until they say it's final (skipped entirely with
    # --skip-review, e.g. for CI). Whatever comes out of this -- AI's
    # original values, or a reviewer's edits -- is what actually gets
    # cached and built into the graph below.
    final_metadata, param_corrections = review.review_or_skip(
        final_metadata, "parameter", args.skip_review, args.review_confidence_threshold,
    )
    corrections_store.append_corrections(param_corrections)

    # Cache the FINAL parameter metadata (post-review), so a future run of
    # this same module reuses the reviewed values without asking again --
    # not just the newly-inferred ones, since a reviewer may have also
    # edited an entry that came from the cache.
    merged_cache = dict(cache_lookup)
    for item in final_metadata:
        merged_cache[(item["ini"][0], item["ini"][1])] = item
    save_cache(module_dir, list(merged_cache.values()))

    filtered_parameter_fields = build_parameter_fields(final_metadata)

    # 4b. Cache Management & AI Inference for output/solution METRICS
    #     (same idea as step 4, but for the JSON keys main.cc writes out,
    #     inferred in SI units rather than picked from KNOWN_METRIC_UNITS.)
    metric_candidates = discover_metrics_from_maincc(main_cc_path)
    metric_keys = [c.key for c in metric_candidates]

    metric_cache = load_metric_cache(module_dir) or []

    # Same staleness pruning as for parameters above, but keyed on the raw
    # metric key against what's currently found in main.cc.
    valid_metric_keys = {c.key for c in metric_candidates}
    pruned_metric_cache = [item for item in metric_cache if item["key"] in valid_metric_keys]
    if len(pruned_metric_cache) != len(metric_cache):
        dropped = len(metric_cache) - len(pruned_metric_cache)
        print(
            f"Pruned {dropped} stale metric cache entr{'y' if dropped == 1 else 'ies'} "
            "no longer present in main.cc.",
            file=sys.stderr,
        )
        save_metric_cache(module_dir, pruned_metric_cache)
    metric_cache = pruned_metric_cache

    metric_cache_lookup = {item["key"]: item for item in metric_cache}

    final_metric_metadata = []
    missing_metric_candidates = []
    for candidate in metric_candidates:
        if candidate.key in metric_cache_lookup:
            final_metric_metadata.append(metric_cache_lookup[candidate.key])
        else:
            missing_metric_candidates.append(candidate)

    if missing_metric_candidates:
        resolved_model = args.model or PROVIDER_CONFIG[args.provider]["default_model"]
        print(f"\nQuerying {args.provider} ({resolved_model}) for "
              f"{len(missing_metric_candidates)} metric(s) not found in cache...")
        known_metric_corrections = corrections_store.relevant_corrections_for(
            [c.key for c in missing_metric_candidates], "metric",
        )
        try:
            new_inferred_metrics = infer_metric_metadata(
                candidates=missing_metric_candidates,
                main_cc=read_text(main_cc_path),
                problem_hh=read_text(problem_hh_path),
                benchmark_description=benchmark_description,
                provider=args.provider,
                model=args.model,
                verbose=args.verbose,
                known_corrections=known_metric_corrections,
            )
        except (RuntimeError, ValueError) as exc:
            if not args.fallback_on_error:
                sys.exit(
                    f"Error: {args.provider} metric inference failed -- {exc}\n"
                    "Pass --fallback-on-error to use local placeholder metadata instead of exiting."
                )
            print(
                f"Warning: {args.provider} metric inference failed ({exc}). "
                f"Using local placeholder metadata for {len(missing_metric_candidates)} metric(s) instead.",
                file=sys.stderr,
            )
            new_inferred_metrics = [
                {
                    "key": candidate.key,
                    "semantic_name": candidate.key,
                    "datatype": "schema:Double",
                    "unit": "unit:UNITLESS",
                    "quantityKind": None,
                    "confidence": 0.0,
                    "explanation": "Placeholder generated because the LLM API call failed.",
                }
                for candidate in missing_metric_candidates
            ]
            final_metric_metadata.extend(new_inferred_metrics)
            # Deliberately NOT written to the cache, same rationale as parameters.
        else:
            final_metric_metadata.extend(new_inferred_metrics)
            # Not saved to the cache yet -- same rationale as parameters:
            # the review step below may still edit these values.
    else:
        print("Using cached metric metadata (loaded from local file, 0 API queries triggered).")

    final_metric_metadata, metric_corrections = review.review_or_skip(
        final_metric_metadata, "metric", args.skip_review, args.review_confidence_threshold,
    )
    corrections_store.append_corrections(metric_corrections)

    merged_metric_cache = dict(metric_cache_lookup)
    for item in final_metric_metadata:
        merged_metric_cache[item["key"]] = item
    save_metric_cache(module_dir, list(merged_metric_cache.values()))

    filtered_metric_fields = build_metric_fields(final_metric_metadata)

    # 5. Process Cases & Build Graph
    manifest = build_manifest(
        read_text(problem_hh_path), read_text(main_cc_path), benchmark_description,
        read_text(readme_path) if readme_path else "",
        read_text(authors_path) if authors_path else "",
        authors_path.name if authors_path else None,
    )
    cases = discover_cases(module_dir)

    builder = GraphBuilder(manifest, filtered_parameter_fields, filtered_metric_fields, benchmark_description)
    metric_ids = builder.ensure_metric_fields(metric_keys)

    config_ids = []
    for case_dir, case_id in cases:
        try:
            params = resolve_case_params(case_dir, filtered_parameter_fields)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        cell_parts = []
        for k, v in params.items():
            name = filtered_parameter_fields[k].get("semantic_name", "")
            if name.lower().startswith("cells_"):
                cell_parts.append(f"{v} cells {name[len('cells_'):]}")
        label = ", ".join(cell_parts) or case_id
        config_ids.append(builder.add_configuration(case_id, label, params))

    builder.add_benchmark_node(config_ids, metric_ids)
    builder.add_rocrate_root()

    doc = {"@context": DEFAULT_CONTEXT, "@graph": builder.graph}
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(builder.graph)} graph nodes, {len(cases)} cases)")

    # Non-RO-Crate build/workflow hints -- collected while scanning the repo
    # (executable name from CMakeLists.txt, module path relative to the repo
    # root) but deliberately NOT part of metadata.jsonld itself, since
    # they're implementation detail rather than semantic metadata. Written
    # as a small sidecar file purely for generate_snakefile.py to consume.
    build_hints = {
        "executable_name": executable_name,
        "module_relative_path": module_relative_path,
    }
    build_hints_path = args.output.with_name(args.output.stem + ".build_hints.json")
    build_hints_path.write_text(json.dumps(build_hints, indent=2), encoding="utf-8")
    print(f"Wrote {build_hints_path}")

    # 6. RO-Crate conformance check -- runs by default so every generated
    #    crate is checked against the official ro-crate-1.1 SHACL profile
    #    before you hand it off. Informational only: never changes the exit
    #    code or touches the file already written above. Skip with
    #    --skip-validation if 'roc-validator' isn't installed / no network.
    if not args.skip_validation:
        validate_rocrate(doc, severity=args.validate_severity)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
