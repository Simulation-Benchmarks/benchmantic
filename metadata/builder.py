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

build() actually writes two documents, not one: the benchmark file itself
(parameters, metrics, processing steps, RO-Crate root -- see
GraphBuilder.add_rocrate_root()) at `args.output`, plus a sidecar dataset
document (author, publisher, software dependencies -- see
GraphBuilder.build_dataset_graph()) at `args.output`'s ".dataset.jsonld"
sibling. describe_benchmark.py renames both into their final
"<benchmark-name>_benchmark.jsonld" / "<benchmark-name>_dataset.jsonld"
names once the benchmark's own name is known.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from ai.cache import (
    cache_path, load_cache, load_metric_cache, load_outputs_config, metric_cache_path,
    outputs_config_path, save_cache, save_metric_cache, save_outputs_config,
)
from ai import corrections as corrections_store
from ai.inference import (
    DEFAULT_BATCH_SIZE, DEFAULT_PROVIDER, DEFAULT_TPM_BUDGET, PROVIDER_CONFIG,
    infer_metric_metadata, infer_parameter_metadata,
)
from ai import review
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
from utils import read_text, slugify, to_number
from metadata import repo_source


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
            # NOTE: dependencies (schema:softwareRequirements) deliberately
            # NOT attached here -- see build_dataset_graph(). They live in
            # the separate {benchmark_name}_dataset.jsonld file alongside
            # author/publisher, not in the benchmark file itself.
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

        Deliberately does NOT attach author/publisher here -- those (plus
        software dependencies) live in a separate document built by
        build_dataset_graph() instead (written to its own
        {benchmark_name}_dataset.jsonld file by describe_benchmark.py), so
        the benchmark file itself stays focused on parameters/metrics/
        processing steps. This doesn't affect RO-Crate 1.1 conformance --
        the profile's root-entity requirements are name/description/
        datePublished/license/hasPart; author and publisher are RECOMMENDED,
        not REQUIRED.
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

    @staticmethod
    def _dataset_guess_note(name_is_guess: bool, has_url: bool) -> str | None:
        if name_is_guess:
            return (
                "Automatically inferred from the repository's hosting domain "
                "(no SPDX copyright header was found in source) -- please verify."
            )
        if has_url:
            return "URL automatically inferred from the repository's hosting domain -- please verify."
        return None

    def build_dataset_graph(self, benchmark_filename: str | None = None) -> list[dict[str, Any]]:
        """Build a standalone {"@graph": [...]} node list -- NOT merged into
        self.graph -- covering the dataset-level provenance that used to
        live on the benchmark file's own root entity: author, publisher,
        and software dependencies (schema:softwareRequirements). Written to
        its own {benchmark_name}_dataset.jsonld file, alongside (but
        separate from) the benchmark file itself, by describe_benchmark.py.

        `benchmark_filename`, if given, is recorded as a plain relative
        "schema:isPartOf" link back to the sibling benchmark file, since the
        two files describe the same benchmark but live in the same output
        directory rather than one crate.

        Mirrors add_rocrate_root()'s old author/publisher assembly logic
        (moved here, not duplicated -- see that method's docstring).
        """
        m = self.manifest
        graph: list[dict[str, Any]] = []

        root_entity: dict[str, Any] = {
            "@id": "./",
            "@type": "schema:Dataset",
            "name": m["label"],
            "description": m.get("root_description") or self.benchmark_description or m["label"],
        }
        if benchmark_filename:
            root_entity["schema:isPartOf"] = {"@id": benchmark_filename}

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
            note = self._dataset_guess_note(m.get("publisher_name_is_guess", False), bool(m.get("publisher_url")))
            if note:
                publisher_node["schema:disambiguatingDescription"] = note
            graph.append(publisher_node)

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
                graph.append(author_node)
                author_refs.append({"@id": author_id})

            # Long AUTHORS/CONTRIBUTORS files are truncated (see
            # extract_authors_list()'s `limit`) -- represent the remainder
            # as one extra node rather than silently dropping them.
            if source == "authors_file" and m.get("authors_omitted"):
                more_id = "local:author_additional_contributors"
                graph.append({
                    "@id": more_id,
                    "@type": "schema:Organization",
                    "name": f"{m['authors_omitted']} additional contributor(s)",
                    "schema:disambiguatingDescription":
                        f"See {m.get('authors_file_name') or 'the authors file'} for the full list.",
                })
                author_refs.append({"@id": more_id})

            root_entity["author"] = author_refs if len(author_refs) > 1 else author_refs[0]

        if m.get("dependencies"):
            root_entity["schema:softwareRequirements"] = [
                _format_dependency(d) for d in m["dependencies"]
            ]

        graph.insert(0, root_entity)
        return graph


# =============================================================================
# 4. Case discovery & Resolution
# =============================================================================



def validate_rocrate(doc: dict[str, Any], severity: str = "REQUIRED", verbose: bool = True) -> bool:
    """Validate `doc` (the full {"@context", "@graph"} document) against the
    RO-Crate 1.1 profile. Prints a completeness percentage plus a pass/fail
    summary (at the requested `severity` threshold); the per-issue detail
    (each failing check's identifier/message/entity) is only printed when
    `verbose` is true -- normal mode just shows the pass/fail line, since
    describe_benchmark.py's final summary already surfaces the outcome.
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

    if all_issues and verbose:
        print(f"\nFailing checks ({len(all_issues)} of {checks['count']} total -- this is the gap behind the "
              f"{100 - completeness:.1f}% incomplete):")
        for issue in all_issues:
            blocking = " [BLOCKS PASS]" if issue.severity >= threshold else ""
            print(f"  [{issue.severity.name}]{blocking} {issue.check.identifier}: {issue.message}", file=sys.stderr)
            if issue.violatingEntity:
                print(f"      entity: {issue.violatingEntity}"
                      + (f"  property: {issue.violatingProperty}" if issue.violatingProperty else ""),
                      file=sys.stderr)
    elif all_issues:
        print(f"({len(all_issues)} check(s) below {severity} severity did not pass -- pass --verbose for detail.)")
    else:
        print("No issues at any severity -- crate is 100% complete against this profile.")

    return passed_at_threshold




# =============================================================================
# 4d. Outputs step -- which artifacts this run should produce.
# =============================================================================
#
# Three keys: "dataset" (provenance sidecar, GraphBuilder.build_dataset_graph()),
# "snakefile" (generated by describe_benchmark.py, outside this module), and
# "description" -- whether the benchmark description (benchmark.jsonld) AND
# its review report are generated at all. "description" defaults True and is
# True for every preset except the dedicated "snakefile-only" one: that's the
# ONLY way to turn it off (see ai.tui's "Workflow only" preset and its
# in-code rationale for why this isn't also an independent Custom checkbox).
# When "description" is False, semantic inference (the Groq/OpenAI call) is
# skipped entirely too -- see _build_impl()'s Infer & review step -- since
# nothing downstream of it (the description, the dataset sidecar, which
# requires the description to exist, and the review report) is being
# generated. describe_benchmark.py reads the resolved selection back out of
# build()'s returned stats dict to decide what else to do.

DEFAULT_OUTPUTS: dict[str, bool] = {"description": True, "dataset": True, "snakefile": True}

_OUTPUT_PRESET_ALIASES: dict[str, dict[str, bool]] = {
    "standard": {"description": True, "dataset": True, "snakefile": True},
    "all": {"description": True, "dataset": True, "snakefile": True},
    "snakefile": {"description": True, "dataset": False, "snakefile": True},
    "snakefile-only": {"description": False, "dataset": False, "snakefile": True},
    "dataset": {"description": True, "dataset": True, "snakefile": False},
    "description-only": {"description": True, "dataset": False, "snakefile": False},
    "none": {"description": True, "dataset": False, "snakefile": False},
}


def parse_outputs_arg(raw: str) -> dict[str, bool]:
    """Parse --outputs' string form: a preset name (see
    _OUTPUT_PRESET_ALIASES -- 'standard'/'all', 'snakefile', 'snakefile-only',
    'dataset', 'description-only'/'none') or an explicit comma-separated list
    of 'dataset'/'snakefile' (e.g. 'dataset,snakefile'). Raises ValueError
    with a message naming what was actually typed on anything else -- the
    CLI caller turns that into a sys.exit(). The review report isn't one of
    the choices here -- it's always generated alongside the benchmark
    description. The comma-separated list form always leaves "description"
    True -- turning it off (which also skips the LLM call, the review
    report, and forces "dataset" off, since the dataset sidecar's
    "schema:isPartOf" link points at the benchmark file) is only reachable
    via the 'snakefile-only' preset, by design; see _OUTPUT_PRESET_ALIASES.
    """
    normalized = raw.strip().lower()
    if normalized in _OUTPUT_PRESET_ALIASES:
        return dict(_OUTPUT_PRESET_ALIASES[normalized])
    keys = {k.strip() for k in normalized.split(",") if k.strip()}
    valid = {"dataset", "snakefile"}
    unknown = keys - valid
    if unknown or not keys:
        presets = ", ".join(sorted(set(_OUTPUT_PRESET_ALIASES) - {"all"}))
        raise ValueError(
            f"--outputs {raw!r} not recognized -- use a preset ({presets}) or a comma-separated "
            f"list of {sorted(valid)}."
        )
    result = {k: (k in keys) for k in valid}
    result["description"] = True
    return result


def _resolve_outputs_selection(args: argparse.Namespace, module_dir: Path) -> dict[str, bool]:
    """Decide which of dataset/snakefile to generate this run (the
    "Outputs" step) -- the review report isn't part of this decision, it's
    always generated regardless. In order:
      1. --outputs, if given explicitly, always wins -- no prompt, no preview.
      2. Otherwise, with a real terminal available and review not skipped,
         ask interactively (curses picker, falling back to a plain-text
         menu) -- seeded from this module's saved selection if one
         exists, or DEFAULT_OUTPUTS otherwise -- then show a preview of
         what that selection will write and confirm with "Generate these
         files? [Y/n]" (_confirm_outputs_preview()) before returning; "n"
         reopens the picker, reseeded with the declined selection, instead
         of proceeding.
      3. Otherwise (CI / --skip-review / no real terminal), silently reuse
         the module's saved selection if one exists, else DEFAULT_OUTPUTS
         -- no prompt, no preview.
    Whatever is resolved is saved back (ai.cache.save_outputs_config) so
    the next run against this module_dir reuses it without asking again.
    """
    saved = load_outputs_config(module_dir)
    if saved is not None:
        # Back-compat: a selection saved by a pre-"description" version of
        # this tool only has "dataset"/"snakefile" -- default the new key
        # to True (the only value every prior release's behavior actually
        # meant) rather than treating it as an implicit opt-out.
        saved.setdefault("description", True)

    if getattr(args, "outputs", None):
        try:
            selection = parse_outputs_arg(args.outputs)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        save_outputs_config(module_dir, selection)
        return selection

    if not args.skip_review:
        current = saved or dict(DEFAULT_OUTPUTS)
        try:
            from ai import tui
        except ImportError:
            tui = None  # type: ignore[assignment]

        # The real benchmark.jsonld filename isn't known yet at this point
        # in the run (it's derived from the graph, which doesn't exist
        # until after inference/build) -- module_dir's own folder name is
        # the best available stand-in, so the Outputs-step preview shows
        # something concrete ("rotatingcylinders_benchmark.jsonld") rather
        # than a generic "<name>" placeholder.
        name_guess = slugify(module_dir.name) or "benchmark"
        filenames = {
            "description": f"{name_guess}_benchmark.jsonld",
            "dataset": f"{name_guess}_dataset.jsonld",
            "snakefile": "Snakefile",
            "review": "review.md",
        }
        while True:
            selection = tui.output_picker(current, filenames) if (tui is not None and tui.available()) else None
            if selection is not None:
                # The curses picker already shows a live, file-by-file
                # preview (and its own "Enter to confirm") before you ever
                # get here -- re-asking "Generate these files? [Y/n]" in
                # plain text on top of that would just be the same
                # question a second time. Only the plain-text fallback
                # (which has no live preview of its own) still needs it.
                break
            selection = _plain_outputs_prompt(current)
            if _confirm_outputs_preview(selection, filenames):
                break
            current = selection  # re-open the prompt seeded with the last choice
        save_outputs_config(module_dir, selection)
        return selection

    selection = saved or dict(DEFAULT_OUTPUTS)
    if saved:
        print(f"Using saved output selection from {outputs_config_path(module_dir).name} "
              "(pass --outputs to override).")
    save_outputs_config(module_dir, selection)
    return selection


def _plain_outputs_prompt(current: dict[str, bool]) -> dict[str, bool]:
    """Plain-text fallback for the Outputs step (no real TTY, curses not
    usable, or the curses picker was cancelled). A numbered preset menu
    first; the last entry ("Custom") drops into a simple toggle loop over
    the two optional items -- same conventions as the plain-text
    parameter selector: type numbers to toggle, Enter to confirm.
    """
    from ai import tui

    preset_names = list(tui.OUTPUT_PRESETS)
    print("\nbenchmantic can produce the following artifacts:\n")
    print("  Benchmark description   (every preset except 'Workflow only' below)")
    print("  Review report           (every preset except 'Workflow only' below)")
    # No "[x]"/"[ ]" marks here -- nothing on this list is toggled from this
    # prompt (a preset is picked below, or "Custom" opens the actual
    # toggle loop further down), so a checkbox-looking mark next to
    # something that isn't clickable here would just read as broken.
    for key, (name, desc) in tui.OUTPUT_ITEM_INFO.items():
        print(f"  {name} -- {desc}")

    print("\n? Choose outputs:")
    for i, name in enumerate(preset_names):
        print(f"  {i}) {name} -- {tui.OUTPUT_PRESET_INFO.get(name, '')}")
    print(f"  {len(preset_names)}) Custom -- {tui.OUTPUT_PRESET_INFO.get('Custom', 'choose individually')}")

    while True:
        choice = input("Selection: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(preset_names):
            return dict(tui.OUTPUT_PRESETS[preset_names[int(choice)]])
        if choice.isdigit() and int(choice) == len(preset_names):
            break
        print("Not a valid choice.")

    keys = list(tui.OUTPUT_ITEM_INFO)
    checked = {k for k in keys if current.get(k)}
    while True:
        print("\nSelect outputs:")
        for i, k in enumerate(keys):
            mark = "x" if k in checked else " "
            print(f"  [{mark}] {i}) {tui.OUTPUT_ITEM_INFO[k][0]}")
        raw = input("Toggle number(s), or Enter to confirm: ").strip().lower()
        if not raw:
            # Custom never turns "description" off -- that's only reachable
            # via the dedicated "Workflow only" preset above (see
            # ai.tui.OUTPUT_PRESETS' comment for why: a Custom combination
            # with dataset=True but description=False would be incoherent,
            # since the dataset sidecar's "schema:isPartOf" link points at
            # the benchmark file).
            result = {k: (k in checked) for k in keys}
            result["description"] = True
            return result
        toggled = [int(t) for t in raw.split(",") if t.strip().isdigit() and 0 <= int(t) < len(keys)]
        if not toggled:
            print("No valid indices recognized.")
            continue
        for i in toggled:
            checked.symmetric_difference_update({keys[i]})


def _confirm_outputs_preview(selection: dict[str, bool], filenames: dict[str, str]) -> bool:
    """Show a file-tree preview of what the resolved Outputs selection will
    write and ask "Generate these files? [Y/n]" before anything is actually
    generated -- the "preview" half of "Outputs step + presets + preview".
    Returns True to proceed as-is, False to go back and reconfigure (the
    caller loops back into the picker, reseeded with this selection).
    Enter/'y'/'Y' confirm; 'n'/'N' decline; anything else re-asks.
    """
    print("\nThis run will generate:\n")
    if selection.get("description", True):
        print("  ✓ Benchmark description")
        print("  ✓ Review report")
    else:
        print("  ✗ Benchmark description (skipped -- Snakefile-only mode, no LLM call)")
        print("  ✗ Review report (skipped -- Snakefile-only mode)")
    for key, (name, _desc) in _import_tui_output_item_info().items():
        mark = "✓" if selection.get(key) else "✗"
        note = f" ({filenames[key]})" if selection.get(key) else " (skipped)"
        print(f"  {mark} {name}{note}")
    while True:
        try:
            answer = input("\n? Generate these files? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return True
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def _import_tui_output_item_info() -> dict[str, tuple[str, str]]:
    """Small indirection so _confirm_outputs_preview() degrades gracefully
    (falls back to raw keys) if ai.tui couldn't be imported for some reason,
    rather than raising out of the Outputs step entirely.
    """
    try:
        from ai import tui
        return tui.OUTPUT_ITEM_INFO
    except ImportError:
        return {"dataset": ("Dataset description", ""), "snakefile": ("Reproducible workflow", "")}


_TOTAL_STEPS = 6


def _print_step_header(step: int, title: str) -> None:
    """"N/6  Title" banner shown once per conceptual pipeline stage
    (Discover/Select parameters/Review/Select outputs/Generate/Validate),
    regardless of --verbose -- these are structural navigation, not
    detail, so they're never hidden.
    """
    bar = "─" * 48
    print(f"\n{bar}")
    print(f"{step}/{_TOTAL_STEPS}  {title}")
    print(bar)


# =============================================================================
# 5. Execution Orchestration
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module_dir", type=str,
                     help="Path to the benchmark module folder (containing main.cc, problem.hh, and "
                          "params.input file(s)), OR a higher-level repo/checkout directory -- the script "
                          "will search recursively for the main.cc+problem.hh pair. Also accepts a "
                          "GitHub/GitLab/any git-reachable URL (https://..., git://..., or an scp-like "
                          "git@host:path spec) -- by default it's cloned into a throwaway temp directory "
                          "that's deleted once this run finishes; see --keep-clone/--ref/--clone-dir/"
                          "--fresh-clone.")
    ap.add_argument("--ref", type=str, default=None,
                     help="Branch, tag, or commit to check out when module_dir is a git URL. Ignored for "
                          "a local module_dir. Default: the remote's default branch.")
    ap.add_argument("--keep-clone", action="store_true",
                     help="If module_dir is a git URL, clone it into a persistent local cache directory "
                          "(repo_source.DEFAULT_CLONE_ROOT, override-able via the BENCHMANTIC_REPO_CACHE "
                          "env var) instead of the default throwaway temp clone -- a later run against the "
                          "same URL reuses it (fetch + checkout in place, not a full re-clone), including "
                          "its .parameter_metadata_cache.json/.metric_metadata_cache.json, so repeat runs "
                          "against the same remote benchmark don't re-query the LLM for everything every "
                          "time. Implied by --clone-dir. Ignored for a local module_dir.")
    ap.add_argument("--clone-dir", type=Path, default=None,
                     help="Explicit directory to clone module_dir into, when it's a git URL, instead of "
                          "either the default throwaway temp clone or --keep-clone's default persistent "
                          "cache location. Implies --keep-clone (an explicit destination is a signal you "
                          "intend to reuse it). Ignored for a local module_dir.")
    ap.add_argument("--fresh-clone", action="store_true",
                     help="With --keep-clone/--clone-dir: if a cached clone already exists at the target "
                          "location, delete it and clone from scratch instead of fetching/checking it out "
                          "in place. Has no effect on the default throwaway clone (already always fresh). "
                          "Ignored for a local module_dir.")
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
                     help="Show what's normally collapsed into short checkmark lines: discovery details "
                          "(resolved module path, README/executable/cache-clear lines) and, per LLM call, "
                          "a one-line request header (endpoint, model, prompt size, estimated cost) plus "
                          "response timing. Doesn't include the raw request/response text -- see --debug.")
    ap.add_argument("--debug", action="store_true",
                     help="Everything --verbose shows, PLUS the exact request/response text for every LLM "
                          "call. Implies --verbose's level of detail even if --verbose isn't also passed.")
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
                     help=f"During review, mark any parameter/metric with confidence below this value with "
                          f"a '!' in the table so it's easy to spot before pressing Enter to accept "
                          f"(default: {review.DEFAULT_CONFIDENCE_THRESHOLD}). Doesn't block accepting on its own -- "
                          "Enter always accepts the table as shown. Has no effect with --skip-review.")
    ap.add_argument("--inference-batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                     help="Max parameters/metrics sent to the LLM in a single inference request "
                          f"(default: {DEFAULT_BATCH_SIZE}). A benchmark with more not-yet-cached items than "
                          "this is split into multiple independent requests instead of one large one. Each "
                          "resulting chunk may be split further still if its own estimated token cost exceeds "
                          "--inference-tpm-budget -- see that flag.")
    ap.add_argument("--inference-tpm-budget", type=int, default=DEFAULT_TPM_BUDGET,
                     help="Target tokens-per-minute budget for the inference step "
                          f"(default: {DEFAULT_TPM_BUDGET:,}). Used two ways: (1) any --inference-batch-size "
                          "chunk whose estimated cost (prompt + expected completion) exceeds this is split "
                          "further, regardless of item count; (2) every request made during this run reserves "
                          "its estimated cost against a shared budget before sending, so a burst of "
                          "individually-small requests can't collectively exceed it either. Set this well "
                          "BELOW your account's actual TPM cap, not equal to it -- token estimation is a "
                          "heuristic (see ai.inference's CHARS_PER_TOKEN_ESTIMATE), and other activity on the "
                          "same API key shares the same rolling window. E.g. on a 12,000 TPM account, try "
                          "8,000-9,000, not 12,000.")
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
    ap.add_argument("--outputs", type=str, default=None,
                     help="Which artifacts to generate. Either a preset -- 'standard' (benchmark "
                          "description + review report + dataset + snakefile, the default), 'snakefile' "
                          "(description + review report + snakefile, no dataset), 'snakefile-only' "
                          "(ONLY a Snakefile -- no semantic inference/LLM call, no benchmark "
                          "description, no dataset, no review report; see snakefile.renderer's "
                          "config_key() docstring for the parameters.json unit-suffix limitation this "
                          "carries), 'dataset' (description + review report + dataset, no snakefile), "
                          "'description-only'/'none' (description + review report only) -- or an "
                          "explicit comma-separated list of 'dataset', 'snakefile' (description stays "
                          "on; 'snakefile-only' is the only way to turn it off). Default: reuse the "
                          "selection saved from this module's last run (.benchmantic_outputs.json), "
                          "prompting interactively (real terminal only) the first time, or 'standard' "
                          "with --skip-review / no real terminal.")
    return ap


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve `module_dir` (a local path, or a git URL -- see
    metadata/repo_source.py), run the actual build (_build_impl()), and clean up a
    throwaway clone afterward regardless of whether the build succeeded or
    raised. getattr() with defaults on the resolve_source() call so this
    also works when `args` comes from a caller (e.g. describe_benchmark.py)
    that didn't set ref/clone_dir/fresh_clone/keep_clone explicitly.

    Returns a small stats dict (parameter/metric/case counts and the
    RO-Crate validation outcome) -- describe_benchmark.py's final summary
    box is built from this, so it doesn't have to re-derive counts by
    re-parsing the JSON-LD it just wrote.
    """
    resolved_module_dir = repo_source.resolve_source(
        str(args.module_dir),
        ref=getattr(args, "ref", None),
        clone_dir=getattr(args, "clone_dir", None),
        fresh_clone=getattr(args, "fresh_clone", False),
        keep_clone=getattr(args, "keep_clone", False),
    )
    args.module_dir = resolved_module_dir
    try:
        return _build_impl(args)
    finally:
        # No-op for a local module_dir or a persistent (--keep-clone/
        # --clone-dir) clone -- only ever removes a throwaway clone this
        # same resolve_source() call created. Runs even if _build_impl()
        # raised, so a failed run doesn't leave an orphaned temp clone.
        repo_source.cleanup_ephemeral_clone(resolved_module_dir)


def _build_impl(args: argparse.Namespace) -> dict[str, Any]:
    if not args.module_dir.is_dir():
        sys.exit(f"Error: {args.module_dir} is not a directory")

    _print_step_header(1, "Discover benchmark")

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
    if args.verbose and module_dir != repo_root:
        print(f"Resolved benchmark module: {module_dir}")

    main_cc_path = module_dir / "main.cc"
    problem_hh_path = module_dir / "problem.hh"

    readme_path = find_readme(module_dir, repo_root)
    if args.verbose and readme_path:
        print(f"Found README: {readme_path}")

    authors_hint = extract_authors_file_hint(read_text(problem_hh_path), read_text(main_cc_path))
    authors_path = find_authors_file(module_dir, repo_root, hint=authors_hint)
    if args.verbose and authors_path:
        print(f"Found authors file: {authors_path}")

    executable_name, cmakelists_path = find_executable_name(module_dir, repo_root)
    if args.verbose and executable_name:
        print(f"Found executable target: {executable_name} (from {cmakelists_path})")
    if not executable_name:
        # Always shown, regardless of verbosity -- this is a warning the
        # reviewer may need to act on (pass --executable explicitly later),
        # not just diagnostic noise.
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
                if args.verbose:
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

    if not args.verbose:
        # Normal mode: one short checklist instead of the individual
        # "Resolved module/Found README/Found executable/Cleared cache"
        # lines above -- those are still available with --verbose.
        print("Discovering benchmark...\n")
        print("  ✓ Source code")
        print(f"  {'✓' if readme_path else '○'} README" + ("" if readme_path else " (not found)"))
        print(f"  {'✓' if executable_name else '○'} CMake target" + ("" if executable_name else " (not found)"))
        print(f"  ✓ Parameters ({len(raw_candidates)} found in params.input)")

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

    _print_step_header(2, "Select parameters")

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

        # Try a curses arrow-key checkbox screen first (see ai.tui) -- only
        # takes over if it's actually usable (a real TTY, curses importable)
        # and the reviewer doesn't back out of it with 'q'/Escape. Any
        # other outcome (no TTY, curses missing, cancelled) falls straight
        # through to the plain-text toggle loop below, completely
        # unaffected -- `checked_indices` is untouched either way.
        curses_handled = False
        try:
            from ai import tui
        except ImportError:
            tui = None  # type: ignore[assignment]
        if tui is not None and tui.available():
            curses_result = tui.checkbox_list(
                "Parameter selection",
                f"{len(raw_candidates)} parameters detected -- {len(checked_indices)} selected "
                f"automatically (under [{sections_label}]).",
                [c.key for c in raw_candidates],
                checked_indices,
            )
            if curses_result is not None:
                checked_indices = curses_result or set(default_indices)
                selected_candidates = [c for i, c in enumerate(raw_candidates) if i in checked_indices]
                curses_handled = True

        # Print tabular structure with checkbox markers, columns sized to
        # the actual terminal width instead of a hardcoded 2 -- a long
        # parameter key (or a narrow terminal) made a fixed 2-column
        # layout wrap mid-row and become unreadable; this drops to 1
        # column whenever 2 wouldn't fit, and would use more than 2 on a
        # very wide terminal if that ever became worth doing (capped at 2
        # for now to keep rows scannable either way).
        longest_formatted_len = max(len(f"[x] {i:2d} {c.key}") for i, c in enumerate(raw_candidates))
        col_width = longest_formatted_len + 4  # padding margin
        term_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        cols = max(1, min(2, term_width // col_width))

        def _print_table() -> None:
            print("\n=== Parameter Selection ===")
            print(f"Parameters under [{sections_label}] are pre-selected by default (marked [x]).\n")
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
            print("  - Type 'table' to reprint this table if you lose track of the selection")
            print("  - Repeat as many times as needed -- each entry toggles from where you left off")
            print("  - Finally, press Enter with no input to confirm the selection and proceed")

        # The full table + instructions are shown once up front. After that,
        # each toggle only prints a one-line summary of what changed rather
        # than reprinting the whole ~38-row table again -- the table is
        # still available on demand via the 'table' command below, but
        # re-showing it after every single edit was the main source of
        # clutter in a multi-toggle session.
        if not curses_handled:
            _print_table()

        while not curses_handled:
            try:
                user_input = input("\nToggle selection (or Enter to confirm): ").strip().lower()

                if not user_input:
                    if not checked_indices:
                        print("No parameters selected. Please select at least one.", file=sys.stderr)
                        continue
                    selected_candidates = [c for i, c in enumerate(raw_candidates) if i in checked_indices]
                    break

                if user_input == "table":
                    _print_table()
                    continue
                if user_input == "all":
                    checked_indices = set(range(len(raw_candidates)))
                    print(f"-> selected all {len(raw_candidates)} parameters ({len(checked_indices)} total).")
                    continue
                if user_input == "none":
                    checked_indices = set()
                    print("-> cleared selection (0 total).")
                    continue

                toggled = [int(tok.strip()) for tok in user_input.split(",") if tok.strip().lstrip("-").isdigit()]
                valid_toggled = [i for i in toggled if 0 <= i < len(raw_candidates)]
                if not valid_toggled:
                    print("No valid indices recognized. Please try again.", file=sys.stderr)
                    continue

                turned_on, turned_off = [], []
                for i in valid_toggled:
                    if i in checked_indices:
                        checked_indices.discard(i)
                        turned_off.append(i)
                    else:
                        checked_indices.add(i)
                        turned_on.append(i)

                parts = []
                if turned_on:
                    parts.append("on: " + ", ".join(f"{i} {raw_candidates[i].key}" for i in turned_on))
                if turned_off:
                    parts.append("off: " + ", ".join(f"{i} {raw_candidates[i].key}" for i in turned_off))
                print(f"-> {'; '.join(parts)}  ({len(checked_indices)} selected total)")

                skipped = len(toggled) - len(valid_toggled)
                if skipped:
                    print(f"   ({skipped} out-of-range index/indices ignored)", file=sys.stderr)

            except (EOFError, KeyboardInterrupt):
                print(
                    f"\nInput cancelled. Falling back to the default: parameters under [{sections_label}].",
                    file=sys.stderr,
                )
                selected_candidates = default_candidates
                break

    if args.verbose:
        print(f"\nFinal Selection Confirmed.")
        print(f"Scenario-Specific: {', '.join(c.key for c in selected_candidates)}")
        print(f"Tool-specific:   {', '.join(c.key for c in raw_candidates if c not in selected_candidates) or 'None'}\n")
    else:
        print(f"\n✓ {len(selected_candidates)}/{len(raw_candidates)} parameters selected")

    # Outputs step moved here (right after parameter selection, before any
    # LLM call) so a reviewer who only wants a Snakefile can say so BEFORE
    # this run pays for semantic inference on every selected parameter --
    # see the "description" key's docstring above _OUTPUT_PRESET_ALIASES.
    _print_step_header(3, "Select outputs")
    outputs_selection = _resolve_outputs_selection(args, module_dir)
    skip_inference = not outputs_selection.get("description", True)

    _print_step_header(4, "Infer & review")
    if skip_inference:
        # Snakefile-only mode: no Groq/OpenAI client is constructed on this
        # path at all -- final_metadata is built directly from the raw
        # params.input candidates, with a cheap static type guess
        # (utils.to_number, the same helper resolve_case_params() itself
        # uses to parse a case's actual values) standing in for the usual
        # LLM-inferred datatype. unit/quantityKind are left unknown
        # ("unit:UNITLESS"/None) rather than guessed -- see config_key()'s
        # "no known unit -> bare parameters.json key" fallback in
        # snakefile/renderer.py, and the module-level warning this prints
        # via describe_benchmark.py right after this step.
        print(
            "(skipped -- Snakefile-only mode: no semantic inference, no Groq/OpenAI call, "
            "no review report. Parameter names/types are guessed statically from params.input; "
            "units are left unknown -- see README.md's Config files section.)"
        )
        final_metadata = [
            {
                "semantic_name": f"{c.section}.{c.key}",
                "ini": [c.section, c.key],
                "index": 0,
                "datatype": _guess_static_datatype(c.tokens[0] if c.tokens else c.value),
                "unit": "unit:UNITLESS",
                "quantityKind": None,
                "confidence": 0.0,
                "explanation": "(no semantic inference -- Snakefile-only mode)",
            }
            for c in selected_candidates
        ]
        final_metric_metadata = []
        metric_keys = []
    else:
        final_metadata, final_metric_metadata, metric_keys = _infer_and_review(
            args, module_dir, selected_candidates, raw_candidates, main_cc_path, problem_hh_path,
            benchmark_description,
        )

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

    filtered_parameter_fields = build_parameter_fields(final_metadata)
    filtered_metric_fields = build_metric_fields(final_metric_metadata) if not skip_inference else {}

    _print_step_header(5, "Generate")

    if skip_inference:
        return _generate_snakefile_only(
            args, module_dir, filtered_parameter_fields, outputs_selection,
            readme_path, executable_name, module_relative_path,
        )

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
    if args.verbose:
        print(f"Wrote {args.output} ({len(builder.graph)} graph nodes, {len(cases)} cases)")

    # Dataset-level provenance (author, publisher, software dependencies) --
    # deliberately a *separate* document from the benchmark file above (see
    # GraphBuilder.build_dataset_graph()'s docstring). Written here to a
    # staged sidecar path next to args.output; describe_benchmark.py renames
    # it to its final "<benchmark-name>_dataset.jsonld" name (and fills in
    # the "schema:isPartOf" link back to the benchmark file's *final* name,
    # which isn't known yet at this point) when it moves both files into
    # outputs/<software-name>/. Only written when "dataset" is part of this
    # run's Outputs selection (see _resolve_outputs_selection() above) --
    # skipped entirely otherwise, so describe_benchmark.py never finds a
    # staged sidecar to move.
    if outputs_selection.get("dataset", True):
        dataset_graph = builder.build_dataset_graph()
        dataset_doc = {"@context": DEFAULT_CONTEXT, "@graph": dataset_graph}
        dataset_staged_path = args.output.with_name(args.output.stem + ".dataset.jsonld")
        dataset_staged_path.write_text(json.dumps(dataset_doc, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"Wrote {dataset_staged_path} ({len(dataset_graph)} graph nodes)")

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
    if args.verbose:
        print(f"Wrote {build_hints_path}")

    # 6. RO-Crate conformance check -- runs by default so every generated
    #    crate is checked against the official ro-crate-1.1 SHACL profile
    #    before you hand it off. Informational only: never changes the exit
    #    code or touches the file already written above. Skip with
    #    --skip-validation if 'roc-validator' isn't installed / no network.
    _print_step_header(6, "Validate")
    rocrate_passed = None
    if not args.skip_validation:
        rocrate_passed = validate_rocrate(doc, severity=args.validate_severity, verbose=args.verbose)
    else:
        print("(skipped -- --skip-validation)")

    if not args.verbose:
        print(f"\n✓ Wrote {args.output.name} ({len(final_metadata)} parameters, "
              f"{len(final_metric_metadata)} metrics, {len(cases)} case(s))")

    return {
        "parameters": len(final_metadata),
        "metrics": len(final_metric_metadata),
        "cases": len(cases),
        "rocrate_validation_passed": rocrate_passed,
        "outputs": outputs_selection,
    }


def _guess_static_datatype(raw_value: str) -> str:
    """Cheap, LLM-free datatype guess for Snakefile-only mode: reuses
    utils.to_number() (the same helper resolve_case_params() itself calls
    to parse a case's actual value from params.input) rather than
    hand-rolling a second int/float/string sniffer -- "int" -> schema:Integer,
    "float" -> schema:Float, anything else -> schema:String.
    """
    n = to_number(raw_value)
    if isinstance(n, int):
        return "schema:Integer"
    if isinstance(n, float):
        return "schema:Float"
    return "schema:String"


def _generate_snakefile_only(
    args: argparse.Namespace,
    module_dir: Path,
    filtered_parameter_fields: dict[str, Any],
    outputs_selection: dict[str, bool],
    readme_path: Path | None,
    executable_name: str | None,
    module_relative_path: str | None,
) -> dict[str, Any]:
    """The Generate step for Snakefile-only mode: no benchmark description,
    no dataset sidecar, no RO-Crate validation -- just enough of a graph
    (case-varying parameter values, plus the "local:software"/"./" nodes
    snakefile.generator.derive_software_name()/derive_build_dir() read) for
    snakefile.generator to build the Snakefile from, written to args.output
    (a STAGING path -- describe_benchmark.py never moves/renames this into
    outputs/<software-name>/ as a real "<name>_benchmark.jsonld" deliverable
    file; see its run()). build_manifest() itself is pure static text
    scraping (SPDX headers, README parsing, ...) with no LLM call, so it's
    still safe/cheap to call here for the software label and repo URL.
    `readme_path`/`executable_name`/`module_relative_path` are reused as-is
    from _build_impl()'s Discover step, not recomputed here.
    """
    problem_hh_path = module_dir / "problem.hh"
    main_cc_path = module_dir / "main.cc"
    manifest = build_manifest(
        read_text(problem_hh_path), read_text(main_cc_path), "",
        read_text(readme_path) if readme_path else "",
    )
    cases = discover_cases(module_dir)
    if not cases:
        sys.exit(f"Error: no benchmark cases found under {module_dir} -- nothing to build a Snakefile from.")

    builder = GraphBuilder(manifest, filtered_parameter_fields)
    config_ids = []
    for case_dir, case_id in cases:
        try:
            params = resolve_case_params(case_dir, filtered_parameter_fields)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        config_ids.append(builder.add_configuration(case_id, case_id, params))

    # Minimal, non-RO-Crate root + software node -- just enough for
    # snakefile.generator's derive_software_name() (local:software.label)
    # and derive_build_dir() (./codeRepository + the build_hints.json
    # sidecar's module_relative_path). Deliberately NOT a real RO-Crate
    # root entity (no ro-crate-metadata.json descriptor, no hasPart, no
    # license/datePublished) -- this graph is only ever consumed by
    # snakefile.generator.generate(), never treated as a deliverable.
    builder.graph.append({"@id": "local:software", "@type": "tool", "label": manifest["software_label"]})
    root_entity: dict[str, Any] = {"@id": "./", "@type": "schema:Dataset", "name": manifest["label"]}
    if manifest.get("repo_url"):
        root_entity["codeRepository"] = manifest["repo_url"]
    builder.graph.insert(0, root_entity)

    doc = {"@context": DEFAULT_CONTEXT, "@graph": builder.graph}
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if args.verbose:
        print(f"Wrote {args.output} (staging only -- Snakefile-only mode, not a real benchmark description)")

    build_hints = {
        "executable_name": executable_name,
        "module_relative_path": module_relative_path,
    }
    build_hints_path = args.output.with_name(args.output.stem + ".build_hints.json")
    build_hints_path.write_text(json.dumps(build_hints, indent=2), encoding="utf-8")

    _print_step_header(6, "Validate")
    print("(skipped -- Snakefile-only mode produces no benchmark.jsonld to validate)")

    if not args.verbose:
        print(f"\n✓ Prepared {len(filtered_parameter_fields)} parameter(s), {len(cases)} case(s) for the Snakefile")

    return {
        "parameters": len(filtered_parameter_fields),
        "metrics": 0,
        "cases": len(cases),
        "rocrate_validation_passed": None,
        "outputs": outputs_selection,
    }


def _infer_and_review(
    args: argparse.Namespace,
    module_dir: Path,
    selected_candidates: list,
    raw_candidates: list,
    main_cc_path: Path,
    problem_hh_path: Path,
    benchmark_description: str,
) -> tuple[list[dict], list[dict], list[str]]:
    """Cache lookup + Groq/OpenAI inference + the combined interactive
    review, for both parameters and metrics. This is the ONLY place in the
    pipeline that ever calls infer_parameter_metadata()/infer_metric_metadata()
    (i.e. the only place an LLM client is constructed) -- a run resolved to
    Snakefile-only mode never calls this function at all, and so needs no
    GROQ_API_KEY/OPENAI_API_KEY set. Returns (final_metadata,
    final_metric_metadata, metric_keys) -- the post-review values that get
    cached and built into the graph.
    """
    # 4a. Cache Management & AI Inference (ONLY for the selected parameters)
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
        # human already fixed once. See ai.corrections / ai.review.
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
                debug=getattr(args, "debug", False),
                known_corrections=known_corrections,
                batch_size=getattr(args, "inference_batch_size", DEFAULT_BATCH_SIZE),
                tpm_budget=getattr(args, "inference_tpm_budget", DEFAULT_TPM_BUDGET),
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

    # NOTE: --full-value-params is applied by the caller (_build_impl), once,
    # to whichever final_metadata this function (or the Snakefile-only
    # branch) produced -- not duplicated here.

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
                debug=getattr(args, "debug", False),
                known_corrections=known_metric_corrections,
                batch_size=getattr(args, "inference_batch_size", DEFAULT_BATCH_SIZE),
                tpm_budget=getattr(args, "inference_tpm_budget", DEFAULT_TPM_BUDGET),
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

    if not args.verbose:
        print(f"\n✓ {len(final_metadata)} parameter(s) analyzed")
        print(f"✓ {len(final_metric_metadata)} metric(s) analyzed")

    # 4c. Interactive review -- parameters AND metrics go through ONE
    # combined review pass here (not two separate back-to-back table
    # sessions), skipped entirely with --skip-review (e.g. for CI).
    # Whatever comes out of this -- AI's original values, or a reviewer's
    # edits -- is what actually gets cached and built into the graph
    # below. See ai.review.review_or_skip_combined()'s docstring for how
    # this picks between a curses queue and the plain-text fallback.
    final_metadata, final_metric_metadata, param_corrections, metric_corrections = review.review_or_skip_combined(
        final_metadata, final_metric_metadata, args.skip_review, args.review_confidence_threshold,
    )
    corrections_store.append_corrections(param_corrections)
    corrections_store.append_corrections(metric_corrections)

    # Cache the FINAL metadata (post-review) for both kinds, so a future
    # run of this same module reuses the reviewed values without asking
    # again -- not just the newly-inferred ones, since a reviewer may have
    # also edited an entry that came from the cache.
    merged_cache = dict(cache_lookup)
    for item in final_metadata:
        merged_cache[(item["ini"][0], item["ini"][1])] = item
    save_cache(module_dir, list(merged_cache.values()))

    merged_metric_cache = dict(metric_cache_lookup)
    for item in final_metric_metadata:
        merged_metric_cache[item["key"]] = item
    save_metric_cache(module_dir, list(merged_metric_cache.values()))

    return final_metadata, final_metric_metadata, metric_keys


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
