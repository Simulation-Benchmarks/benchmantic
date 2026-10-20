# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats.python_code_runtime

The part of the `python` input format that the generated workflow needs
too: this file is copied as it is into Snakefiles that run a Python script
or notebook, so it uses the standard library only and nothing else of
benchmantic.

  code_view(text, notebook)   a script's code; a notebook's code cells joined
                              (shell/magic lines like "%matplotlib" commented out)
  constants(code)             the top-level constants: `r = 0.05`, `dt = 1 / 1600`,
                              `points = (0.15, 0.2)`, or the one number inside a
                              call, as in `mu = Constant(mesh, PETSc.ScalarType(0.001))`
  set_constants(code, values) the code with new values for some constants
  metrics_footer(exprs, path) lines that write the values of `exprs` (Python
                              expressions over the script's variables, e.g.
                              "max(C_D)") as JSON to `path` when the script ends
"""

import ast
import json
import operator

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def code_view(text, notebook=False):
    """The code of a script, or of a notebook's code cells, each cell
    headed by "# %% [cell N]". Lines starting with % or ! (IPython magics
    and shell commands) are commented out, so the result runs with plain
    python."""
    if not notebook:
        return text
    cells = json.loads(text).get("cells", [])
    parts = []
    for number, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        lines = [("# " + line) if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()]
        parts.append(f"# %% [cell {number}]\n" + "\n".join(lines))
    return "\n\n".join(parts) + "\n"


def number_value(node):
    """The value of a numeric constant expression (numbers, + - * / // % **,
    parentheses), or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = number_value(node.operand)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = number_value(node.left), number_value(node.right)
        if left is None or right is None:
            return None
        try:
            return _BINARY[type(node.op)](left, right)
        except (ArithmeticError, ValueError):
            return None
    return None


def _numbers(node):
    """[value] for a number, [v1, v2, ...] for a tuple/list of numbers, else None."""
    value = number_value(node)
    if value is not None:
        return value
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        values = [number_value(e) for e in node.elts]
        if all(v is not None for v in values):
            return values
    return None


def _numbers_in_call(call):
    """The numeric arguments inside a call (also inside calls passed as
    arguments), as (node, value) pairs -- or None when an argument is
    computed from something else (then the call is not a constant)."""
    found = []

    def visit(node):
        values = _numbers(node)
        if values is not None:
            found.append((node, values))
            return True
        if isinstance(node, ast.Call):
            # numbers in keyword arguments (figsize=(25, 8), shape=...) are options, not inputs
            return all(visit(a) for a in node.args) and all(_numbers(k.value) is not None or visit(k.value)
                                                            for k in node.keywords)
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(visit(e) for e in node.elts)
        # names, attributes, strings: not numbers, but no computation either
        return isinstance(node, (ast.Name, ast.Attribute)) or (
            isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)))

    return found if visit(call) else None


def _top_level(body):
    """Statements at the top level, also inside top-level if/with/try
    blocks (e.g. `if __name__ == "__main__":`), not in loops or functions."""
    for statement in body:
        yield statement
        if isinstance(statement, (ast.If, ast.With)):
            yield from _top_level(statement.body)
            yield from _top_level(getattr(statement, "orelse", []))
        elif isinstance(statement, ast.Try):
            for block in (statement.body, statement.orelse, statement.finalbody):
                yield from _top_level(block)


def _receiver(value):
    """`obj` for a call obj.method(...) / obj.a.method(...), else None."""
    func = value.func if isinstance(value, ast.Call) else None
    if not isinstance(func, ast.Attribute):
        return None
    node = func.value
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def constants(code):
    """The top-level constants of `code`, in order, one per name (the first
    assignment of a name counts). Each is a dict: name, value (a number or
    a list of numbers), line (1-based), the span of the value in the code
    -- start/end line and column, 0-based columns -- and `wrapped` (True
    when the number sits inside a call)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found, seen, assigned = [], set(), set()
    for statement in _top_level(tree.body):
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        computed_on = _receiver(value)
        assigned.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        if computed_on in assigned - seen:
            continue      # a method of something the code computed: cells.links(0) is an index, not an input
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names or len(names) != len(targets):
            continue
        numbers = _numbers(value)
        wrapped = False
        node = value
        if numbers is None and isinstance(value, ast.Call):
            inner = _numbers_in_call(value)
            uses_inputs = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)} & seen
            if inner and len(inner) == 1 and not uses_inputs:   # not mixed with other inputs
                node, numbers = inner[0]
                wrapped = True
        if numbers is None:
            continue
        for name in names:
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)
            found.append({"name": name, "value": numbers, "line": statement.lineno,
                          "start": (node.lineno, node.col_offset), "end": (node.end_lineno, node.end_col_offset),
                          "wrapped": wrapped, "is_list": isinstance(numbers, list),
                          "list_type": "[]" if isinstance(node, ast.List) else "()"})
    return found


def _literal(text, constant):
    """`text` (a value as benchmantic passes it: "0.05", or "1.0 1.5" for
    several numbers) as Python source for this constant."""
    text = str(text).strip()
    if constant["is_list"]:
        items = [t for t in text.replace(",", " ").replace("(", " ").replace(")", " ")
                 .replace("[", " ").replace("]", " ").split() if t]
        body = ", ".join(items) + ("," if len(items) == 1 and constant["list_type"] == "()" else "")
        return constant["list_type"][0] + body + constant["list_type"][1]
    try:
        float(text)
        return text
    except ValueError:
        return repr(text)


def _offset(lines, line, column):
    return sum(len(x) for x in lines[:line - 1]) + column


def set_constants(code, values):
    """`code` with the constants named in `values` ({name: value}) set to
    those values. Only the number itself is replaced, so a constant written
    inside a call keeps the call. Unknown names are an error."""
    by_name = {c["name"]: c for c in constants(code)}
    missing = [n for n in values if n not in by_name]
    if missing:
        raise KeyError(f"no top-level constant named {', '.join(missing)} in the code")
    lines = code.splitlines(keepends=True)
    edits = []
    for name, value in values.items():
        c = by_name[name]
        # ast columns are UTF-8 byte offsets: convert them to characters
        start = _offset(lines, c["start"][0], len(lines[c["start"][0] - 1].encode()[:c["start"][1]].decode()))
        end = _offset(lines, c["end"][0], len(lines[c["end"][0] - 1].encode()[:c["end"][1]].decode()))
        edits.append((start, end, _literal(value, c)))
    for start, end, text in sorted(set(edits), reverse=True):
        code = code[:start] + text + code[end:]
    return code


def metrics_footer(expressions, path):
    """Python lines to append to a script: when it ends, evaluate each
    expression ({metric key: expression}) in the script's namespace and
    write {key: value} as JSON to `path` (on MPI rank 0 only)."""
    return f'''

# --- added by benchmantic: write the benchmark metrics ---
def _benchmantic_write_metrics():
    import json as _json
    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.rank != 0:
            return
    except ImportError:
        pass
    _values = {{}}
    for _key, _expression in {dict(expressions)!r}.items():
        try:
            _value = eval(_expression, globals())
            _values[_key] = float(getattr(_value, "real", _value))
        except Exception as _error:
            print(f"benchmantic: metric {{_key}} = {{_expression}} failed: {{_error}}")
            _values[_key] = None
    with open({str(path)!r}, "w") as _file:
        _json.dump(_values, _file, indent=2)


_benchmantic_write_metrics()
'''
