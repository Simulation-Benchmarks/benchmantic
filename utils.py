"""Generic helpers with no domain coupling -- string/number/file utilities
used across the metadata, ai, and snakefile packages.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path


def read_text(path: Path | None) -> str:
    """Read a file's text, or "" if it's None or doesn't exist."""
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_dune_ini(path: Path) -> dict[str, dict[str, str]]:
    """Parse a DUNE/DuMux params.input file into {section: {key: value}}."""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.optionxform = str
    text = path.read_text(encoding="utf-8")
    cp.read_string(text)
    return {section: dict(cp.items(section)) for section in cp.sections()}


def to_number(token: str) -> float | int | str:
    """Convert a params.input token to int/float if possible, else leave as-is."""
    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        return float(token)
    except ValueError:
        return token


def slugify(value) -> str:
    """Make a value safe for use inside a JSON-LD @id, e.g. '1.1' -> '1p1'."""
    s = str(value)
    s = s.replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def camel_to_label(name: str) -> str:
    """'RotatingCylindersProblem' -> 'rotating cylinders problem'."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return spaced.strip().lower()
