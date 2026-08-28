# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.tui

Optional curses-based interactive UI for the parameter selector
(metadata.builder's interactive parameter-selection loop) and the combined
semantic-review queue (a single pass over every parameter/metric that
needs a look, replacing ai.review's two separate plain-text tables), used
in place of the plain input()/print() prompts when the terminal supports
it.

Deliberately just a thin, best-effort layer on top of the existing
plain-text flows, not a replacement for them: every entry point here
either returns a real result, or returns None to mean "couldn't do this,
fall back to the plain-text equivalent" -- it never raises out to the
caller. "Couldn't do this" covers a lot of cases on purpose:
  - stdin/stdout isn't a real TTY (piped, redirected, CI, --skip-review
    runs, a test harness)
  - the `curses` module isn't importable at all (e.g. bare Windows
    without a separate `windows-curses` install)
  - the terminal is too small, or its terminfo entry does something
    unexpected mid-draw
  - the user deliberately backs out with 'q'/Escape -- a real escape
    hatch, not just an error path, for anyone who'd rather use the plain
    prompt
  - anything else goes wrong inside curses -- caught broadly and
    defensively, same rationale as ai.validation's network fallbacks:
    worst case is exactly today's plain-text behavior, this only ever
    adds a nicer path on top when it works.

Callers should always be written as:

    result = tui.checkbox_list(...)
    if result is None:
        <run the plain-text equivalent instead>

Field-edit validation (what counts as a valid unit/datatype/quantityKind,
and the bookkeeping an edit triggers -- confidence reset, correction note,
"_edited" marker) is NOT duplicated here: both this module's review queue
and ai.review's plain-text row editor call the same
ai.review._apply_field_edit() so the two front-ends can't drift apart on
what an "edit" actually does to an item.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

try:
    import curses
except ImportError:  # pragma: no cover -- e.g. bare Windows without windows-curses
    curses = None  # type: ignore[assignment]

if curses is not None:
    # ncurses waits up to ESCDELAY ms (1000 by default) after a bare ESC
    # byte before deciding it isn't the start of an arrow-key/function-key
    # escape sequence -- over a laggy terminal (SSH, tmux, some macOS
    # terminal apps) the remaining bytes of an arrow-key sequence can
    # arrive juuust after that window closes, so the keypress is silently
    # dropped or misread as a bare Escape (which every screen here treats
    # as "cancel"). That reads as "arrow keys / toggling / editing don't
    # work" even though the code path itself is fine. Only the ESCDELAY
    # env var (read once, at initscr() time -- i.e. inside curses.wrapper()
    # below, which we don't control the timing of) reliably fixes this;
    # curses.set_escdelay() only works if called *before* initscr, which
    # would mean reaching into wrapper()'s internals. setdefault() so an
    # operator's own ESCDELAY (if set) is still respected.
    os.environ.setdefault("ESCDELAY", "25")

from ai.review import (
    ALLOWED_DATATYPES,
    METRIC_FIELDS,
    PARAM_FIELDS,
    _apply_field_edit,
    _append_note,
    _confidence,
    _label,
)

ENTER_KEYS = (10, 13)


def available() -> bool:
    """Whether it's worth even trying a curses UI: the module imported,
    and both stdin and stdout look like a real interactive terminal.
    """
    if curses is None:
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


#: Purpose-first labels for the Outputs step (see output_picker() below) --
#: keyed the same way metadata.builder's outputs selection dict is, so
#: callers can zip the two together directly. "benchmark" (the semantic
#: description itself) and "review" (the review report) are deliberately
#: NOT in here: both are always generated regardless of what's chosen
#: here (every other artifact either derives from or references the
#: benchmark description; the review report is produced alongside it the
#: same way), so both are shown as locked/always-on lines rather than a
#: real choice -- only "dataset" and "snakefile" are actual toggles.
OUTPUT_ITEM_INFO = {
    "dataset": ("Dataset description", "Author, publisher, and software-dependency provenance."),
    "snakefile": ("Reproducible workflow", "Snakemake workflow for executing the benchmark."),
}

#: Outputs-step presets (see metadata.builder's Outputs step) -- "Custom"
#: isn't listed here, it's handled as its own branch in output_picker()
#: since it opens the checkbox screen instead of returning a fixed dict.
#: The full 2x2 grid over the two real toggles (dataset, snakefile) is
#: covered exactly once each: both on ("Standard"), snakefile only
#: ("Snakefile"), dataset only ("With dataset"), neither ("Description
#: only").
OUTPUT_PRESETS: dict[str, dict[str, bool]] = {
    "Standard": {"dataset": True, "snakefile": True},
    "Snakefile": {"dataset": False, "snakefile": True},
    "Description only": {"dataset": False, "snakefile": False},
    "With dataset": {"dataset": True, "snakefile": False},
}

#: One-line, plain-language description of what each preset actually
#: does, shown next to its name wherever a preset is offered (the curses
#: picker below and metadata.builder's plain-text fallback) instead of
#: just a comma list of included artifact keys. The semantic description
#: (benchmark.jsonld) AND the review report are always generated no
#: matter which preset is chosen -- neither is one of the two toggles
#: these presets set -- so every description below leads with that
#: shared fact, then names whatever else (dataset provenance and/or a
#: Snakefile) that preset adds on top of it.
OUTPUT_PRESET_INFO: dict[str, str] = {
    "Standard": (
        "Generates the semantic description and review report for the corresponding benchmark, "
        "plus dataset provenance and a reproducible Snakemake workflow."
    ),
    "Snakefile": (
        "Generates the semantic description and review report for the corresponding benchmark, "
        "plus a reproducible Snakemake workflow -- no separate dataset-provenance file."
    ),
    "Description only": (
        "Generates the semantic description for the corresponding benchmark and a review report "
        "covering only its input and output parameters -- no dataset provenance, no Snakefile."
    ),
    "With dataset": (
        "Generates the semantic description for the corresponding benchmark, plus dataset "
        "provenance and a review report -- no Snakefile."
    ),
    "Custom": "Choose individually whether to also generate dataset provenance and/or a Snakefile.",
}


class _Cancelled(Exception):
    """Raised internally when the reviewer presses 'q'/Escape to back out
    of a curses screen on purpose. Caught at the same outer boundary as
    any other curses failure -- both mean "fall back to plain text" to
    the caller, so there's no separate signal needed beyond returning
    None either way.
    """


def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """curses raises if a write would spill past the window's
    bottom-right corner -- clip defensively instead of crashing the whole
    UI over a narrow terminal or a long label."""
    try:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        win.addstr(y, x, text[: max(0, max_x - x - 1)], attr)
    except curses.error:
        pass


def _wrap_text(text: str, width: int) -> list[str]:
    """Greedy word-wrap: splits `text` into lines no wider than `width`.
    Used anywhere a piece of text can't be bounded in advance (an
    explanation, a preset description, a "current value" pulled from an
    item) so it's shown in full across multiple lines instead of being
    silently clipped mid-word by _safe_addstr on anything narrower than a
    maximized terminal window. Falls back to a hard character-break for a
    single word longer than `width` on its own, so this can never loop
    forever or crash on pathological input; always returns at least one
    (possibly empty) line.
    """
    if width <= 0:
        return [text]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while len(word) > width:
            lines.append(word[:width])
            word = word[width:]
        current = word
    if current:
        lines.append(current)
    return lines or [""]


def _pretty(value: str | None, prefix: str) -> str:
    """"schema:Float" -> "Float", "unit:DEG" -> "DEG", None -> "" -- just
    for display, never written back to an item."""
    if not value:
        return ""
    return value[len(prefix):] if value.startswith(prefix) else value


def _select_menu(
    stdscr, y0: int, title: str, options: list[str], start: int = 0, header=None,
) -> int | None:
    """Draw a simple arrow-key single-select list starting at row `y0` and
    return the chosen option's index, or None if the reviewer cancelled
    ('q'/Escape).

    Up/Down (or j/k) move the highlight, and for up to the first 9
    options pressing its digit (shown as a "N) " prefix) jumps the
    highlight straight there too -- a way to reach an item that doesn't
    depend on arrow-key escape sequences being recognized at all, for a
    terminal/connection where those are unreliable (see the ESCDELAY note
    above). Either way, the highlighted option is only actually chosen
    once Enter is pressed -- a stray digit or arrow keypress just moves
    the highlight (visibly, right away) rather than committing anything,
    so there's always a chance to see where you landed before confirming.

    The whole screen is erased and redrawn every frame (not just the menu
    itself) -- `header`, if given, is called right after the erase to
    redraw whatever context (a boxed item panel, an artifact summary)
    the caller wants to keep showing above the menu. Without this, a
    terminal resize mid-menu (curses reports a new width on the very next
    getch()) would leave stale characters from the old width's layout
    sitting underneath the newly-positioned text -- exactly the kind of
    overlapping, half-overwritten text a plain per-field addstr with no
    erase produces. Redrawing everything from scratch every frame avoids
    that regardless of whether or when a resize happens.
    """
    cursor = max(0, min(start, len(options) - 1))
    while True:
        stdscr.erase()
        if header is not None:
            header()
        _safe_addstr(stdscr, y0, 0, title, curses.A_BOLD)
        for i, opt in enumerate(options):
            attr = curses.A_REVERSE if i == cursor else 0
            marker = "❯ " if i == cursor else "  "
            digit = f"{i + 1}) " if i < 9 else "   "
            _safe_addstr(stdscr, y0 + 1 + i, 0, f"{marker}{digit}{opt}", attr)
        _safe_addstr(
            stdscr, y0 + 1 + len(options) + 1, 0,
            "  ↑/↓ or a number to highlight, enter to confirm, q cancel",
        )
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(options)
        elif ord("1") <= key <= ord("9") and (key - ord("1")) < len(options):
            cursor = key - ord("1")
        elif key in ENTER_KEYS:
            return cursor
        elif key in (27, ord("q"), ord("Q")):
            return None


def _select_menu_with_details(
    stdscr, y0: int, title: str, options: list[str], details: list[str], start: int = 0, header=None,
) -> int | None:
    """Like _select_menu, but each option can carry an extra line of
    detail text (a preset's full description, say) shown word-wrapped and
    indented beneath its name -- used wherever an option's label alone
    doesn't fit the story in a few words. Plain _select_menu concatenates
    a description onto the same line as the option name; that's fine for
    a short one, but the Outputs step's preset descriptions are full
    sentences that would get silently clipped mid-word on anything
    narrower than a very wide terminal. Same controls as _select_menu
    (arrows/j/k or a digit 1-9 moves the highlight, visibly, right away;
    Enter confirms whatever's highlighted; q/Escape cancels) -- only the
    layout differs, and only the option's own name line highlights/
    responds to the cursor, not its detail lines.

    Like _select_menu, the screen is erased and fully redrawn every frame
    (`header`, if given, redraws whatever the caller wants above the
    menu) so a terminal resize mid-menu can't leave stale text from the
    previous width's line-wrapping sitting underneath the new layout.
    """
    cursor = max(0, min(start, len(options) - 1))
    while True:
        stdscr.erase()
        if header is not None:
            header()
        max_y, max_x = stdscr.getmaxyx()
        detail_width = max(10, max_x - 7)
        wrapped = [_wrap_text(d, detail_width) if d else [] for d in details]
        _safe_addstr(stdscr, y0, 0, title, curses.A_BOLD)
        y = y0 + 1
        for i, opt in enumerate(options):
            attr = curses.A_REVERSE if i == cursor else 0
            marker = "❯ " if i == cursor else "  "
            digit = f"{i + 1}) " if i < 9 else "   "
            _safe_addstr(stdscr, y, 0, f"{marker}{digit}{opt}", attr)
            y += 1
            for line in wrapped[i]:
                _safe_addstr(stdscr, y, 6, line)
                y += 1
        _safe_addstr(stdscr, y + 1, 0, "  ↑/↓ or a number to highlight, enter to confirm, q cancel")
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(options)
        elif ord("1") <= key <= ord("9") and (key - ord("1")) < len(options):
            cursor = key - ord("1")
        elif key in ENTER_KEYS:
            return cursor
        elif key in (27, ord("q"), ord("Q")):
            return None


def _text_input(stdscr, y: int, prompt: str, initial: str = "") -> str | None:
    """Text prompt drawn starting at row `y`, using curses' own line-edit
    (echo + getstr) rather than dropping out of curses mode -- keeps the
    whole interaction inside the same screen. Returns the typed text
    (stripped), or None if the reviewer cancelled with an empty
    Escape-then-Enter... in practice curses' getstr() doesn't expose
    Escape directly, so cancellation here is just "typed nothing" (an
    empty string counts as "no change", same as the plain-text editor).

    `prompt` (e.g. "New value for explanation (current: '...a full
    sentence...'): ") is word-wrapped across as many lines as it needs to
    fit the window's actual width -- it used to be a single _safe_addstr
    call that silently CLIPPED anything past the window edge, which for a
    field with a long current value (explanations routinely run well past
    100 characters) meant most of it was simply invisible on anything
    short of a maximized terminal, even after the input line itself was
    fixed to always have room to type into (see below). Typing always
    happens on its own line, right after however many lines the prompt
    took, starting at a fixed column -- this used to put the input cursor
    at `len(prompt) + 1`, clamped to `max_x - 1` if that overflowed, which
    for the same long-current-value case pinned the cursor at the very
    last column of the screen with no room left to type into at all
    ("the cursor doesn't move" / "I can't edit"). Decoupling the input
    line from the prompt's length (now its *line count*, not raw
    character count) is what keeps both problems from recurring
    regardless of how long a field's current value is.
    """
    max_y, max_x = stdscr.getmaxyx()
    wrap_width = max(10, max_x - 1)
    prompt_lines = _wrap_text(prompt, wrap_width)
    # Reserve at least 1 line for the input row below the (possibly
    # multi-line) prompt; if the prompt alone would eat the whole window,
    # clip the line COUNT (not each line's text) so there's always
    # somewhere left to type.
    max_prompt_lines = max(1, max_y - 1)
    prompt_lines = prompt_lines[:max_prompt_lines]
    prompt_y = min(y, max(0, max_y - 1 - len(prompt_lines)))
    for i, line in enumerate(prompt_lines):
        _safe_addstr(stdscr, prompt_y + i, 0, line)
        stdscr.clrtoeol()
    input_y = min(prompt_y + len(prompt_lines), max_y - 1)
    _safe_addstr(stdscr, input_y, 0, "> ")
    stdscr.clrtoeol()
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    try:
        n = max(1, min(200, max_x - 3))
        raw = stdscr.getstr(input_y, 2, n)
        text = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        text = ""
    finally:
        curses.noecho()
        curses.curs_set(0)
    return text or None


# =============================================================================
# Parameter selector
# =============================================================================

def checkbox_list(title: str, subtitle: str, labels: list[str], checked: set[int]) -> set[int] | None:
    """Full-screen arrow-key checkbox list -- the curses counterpart to
    metadata.builder's plain-text "type index numbers to toggle" loop.
    Up/Down (or j/k) move, Space toggles the highlighted row, 'a' checks
    everything, 'n' clears everything, Enter confirms and returns the
    final checked-index set, 'q'/Escape cancels and returns None (the
    caller falls back to the plain-text prompt with `checked` untouched).
    """
    if curses is None or not labels:
        return None
    try:
        return curses.wrapper(_checkbox_list_impl, title, subtitle, labels, set(checked))
    except _Cancelled:
        return None
    except Exception:
        return None


def _checkbox_list_impl(stdscr, title: str, subtitle: str, labels: list[str], checked: set[int]) -> set[int]:
    curses.curs_set(0)
    stdscr.keypad(True)
    cursor, top = 0, 0
    # A typed number buffer for jumping straight to (and toggling) any row
    # by its absolute position, not just the first 9 currently on screen --
    # a 38-parameter list has rows well past 9, and scrolling to a row
    # before you could type its digit defeated the point. Digits
    # accumulate here until Enter commits them; every other key clears it
    # first, same convention as a typeahead buffer in any line editor.
    # As each digit is typed, the highlight jumps live to that row (so
    # far) -- not just after Enter -- and the digits themselves are shown
    # on their own dedicated line (with a real, visible cursor), never
    # sharing a line with other text that could push them off-screen.
    buffer = ""
    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)
        _safe_addstr(stdscr, 1, 0, subtitle)
        footer_width = max(10, max_x - 1)
        # A one-line "key: action" legend for how to move/select sits right
        # under the title/subtitle, so it's the first thing visible rather
        # than something to scroll past a long list to find. State that
        # changes as you interact -- the running selected count, the
        # in-progress "Go to #:" buffer, and the less-frequently-needed
        # a/n/q shortcuts -- lives right after the list instead, closer to
        # where your eye already is once you've scanned through it. Every
        # group is independently word-wrapped to the terminal's actual
        # width, so a narrow terminal spills a group onto another line of
        # its own rather than silently clipping an option's own text.
        nav_lines = _wrap_text("↑/↓: move, space: toggle, type a number: to jump", footer_width)
        y = 2
        for line in nav_lines:
            _safe_addstr(stdscr, y, 0, line)
            y += 1
        list_top = y + 1  # one blank row between the nav legend and the list

        status_lines = _wrap_text(f"{len(checked)}/{len(labels)} selected", footer_width)
        action_lines = _wrap_text(
            "enter: toggle row, or finish if empty, a: all, n: none, q: cancel", footer_width,
        )
        # 1 blank row + status_lines + the "Go to #:" line + action_lines
        reserved_bottom = 1 + len(status_lines) + 1 + len(action_lines)
        visible_rows = max(1, max_y - list_top - reserved_bottom)
        if cursor < top:
            top = cursor
        if cursor >= top + visible_rows:
            top = cursor - visible_rows + 1
        digit_width = len(str(len(labels)))
        for row, i in enumerate(range(top, min(len(labels), top + visible_rows))):
            mark = "x" if i in checked else " "
            attr = curses.A_REVERSE if i == cursor else 0
            _safe_addstr(stdscr, list_top + row, 0, f"[{mark}] {i + 1:>{digit_width}}) {labels[i]}", attr)

        y = list_top + visible_rows + 1
        for line in status_lines:
            _safe_addstr(stdscr, y, 0, line)
            y += 1
        goto_y = y
        _safe_addstr(stdscr, goto_y, 0, f"Go to #: {buffer}", curses.A_BOLD if buffer else 0)
        stdscr.clrtoeol()
        y += 1
        for line in action_lines:
            _safe_addstr(stdscr, y, 0, line)
            y += 1
        if buffer:
            curses.curs_set(1)
            with contextlib.suppress(curses.error):
                stdscr.move(goto_y, len("Go to #: ") + len(buffer))
        else:
            curses.curs_set(0)
        stdscr.refresh()

        key = stdscr.getch()
        if ord("0") <= key <= ord("9"):
            if len(buffer) < digit_width:
                buffer += chr(key)
                idx = int(buffer) - 1
                if 0 <= idx < len(labels):
                    cursor = idx
            continue
        if key in (curses.KEY_BACKSPACE, 127, 8):
            buffer = buffer[:-1]
            if buffer:
                idx = int(buffer) - 1
                if 0 <= idx < len(labels):
                    cursor = idx
            continue
        if key in ENTER_KEYS:
            if buffer:
                idx = int(buffer) - 1
                buffer = ""
                if 0 <= idx < len(labels):
                    cursor = idx
                    checked.symmetric_difference_update({idx})
                continue
            return checked
        if key in (27, ord("q"), ord("Q")):
            if buffer:
                # Abort just the in-progress number, not the whole screen
                # -- a typo while typing an index shouldn't cost the
                # reviewer their place in the checklist.
                buffer = ""
                continue
            raise _Cancelled()
        buffer = ""
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(labels)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(labels)
        elif key == ord(" "):
            checked.symmetric_difference_update({cursor})
        elif key in (ord("a"), ord("A")):
            checked = set(range(len(labels)))
        elif key in (ord("n"), ord("N")):
            checked = set()


# =============================================================================
# Combined semantic-review queue
# =============================================================================

_ACTIONS = ["Accept", "Rename", "Change unit", "Change type", "Change quantity kind", "Edit explanation", "Skip"]
_FIELD_BY_ACTION = {
    "Rename": "semantic_name",
    "Change unit": "unit",
    "Change type": "datatype",
    "Change quantity kind": "quantityKind",
    "Edit explanation": "explanation",
}


def review_queue(
    flagged: list[tuple[str, dict]], accepted: list[tuple[str, dict]], threshold: float
) -> bool | None:
    """Curses-driven combined review queue: shows a summary of what's
    flagged -- low confidence, or a structural "_needs_verification" flag
    (see ai.review._flagged_indices()) -- versus what was auto-accepted,
    across BOTH parameters and metrics in one pass, instead of two
    separate plain-text tables shown one after another. The reviewer
    explicitly chooses whether to walk through just the flagged items, the
    flagged items PLUS the auto-accepted ones too (in case they'd rather
    double-check everything, not just what the confidence heuristic
    flagged), or none of it.

    `flagged` and `accepted` are each lists of (kind, item) pairs where
    kind is "parameter" or "metric"; every item is mutated in place
    through ai.review._apply_field_edit(), exactly like the plain-text
    editor does, so the caller's downstream cache-save/correction-recording
    code doesn't need to know which UI produced the edits.

    Returns True once the reviewer's chosen set has been walked through
    (accepted, edited, or skipped -- all three just advance to the next
    one; the difference only affects what note gets appended) or they
    chose to accept everything without reviewing, or None if curses isn't
    usable or the reviewer cancelled outright -- in which case nothing has
    been touched by this function (individual items may still carry edits
    from *before* a cancel, since editing happens in place item-by-item,
    not in one atomic commit at the end) and the caller should fall back
    to ai.review.interactive_review() for both kinds sequentially.
    """
    if curses is None or (not flagged and not accepted):
        return None
    try:
        return bool(curses.wrapper(_review_queue_impl, flagged, accepted, threshold))
    except _Cancelled:
        return None
    except Exception:
        return None


def _draw_summary(stdscr, flagged: list[tuple[str, dict]], accepted: list[tuple[str, dict]]) -> None:
    stdscr.erase()
    _safe_addstr(stdscr, 0, 0, "Semantic review", curses.A_BOLD)
    _safe_addstr(stdscr, 2, 0, f"✓ {len(accepted)} accepted automatically")
    _safe_addstr(stdscr, 3, 0, f"! {len(flagged)} require your review")
    y = 5
    for i, (kind, item) in enumerate(flagged, start=1):
        _safe_addstr(stdscr, y, 2, f"{i}. {_label(item, kind)}  ({item.get('semantic_name', '')})")
        y += 1
    y += 1
    if flagged:
        _safe_addstr(stdscr, y, 0, f"  [F] Review the {len(flagged)} flagged item(s)  (or press Enter)")
        y += 1
    if accepted:
        verb = "Also review" if flagged else "Review"
        _safe_addstr(stdscr, y, 0, f"  [A] {verb} the {len(accepted)} auto-accepted item(s)")
        y += 1
    _safe_addstr(stdscr, y, 0, "  [N] Accept everything as-is, no review")
    y += 1
    _safe_addstr(stdscr, y, 0, "  [Q] Quit to plain text")


def _review_queue_impl(stdscr, flagged: list[tuple[str, dict]], accepted: list[tuple[str, dict]], threshold: float) -> bool:
    curses.curs_set(0)
    stdscr.keypad(True)

    _draw_summary(stdscr, flagged, accepted)
    stdscr.refresh()
    queue: list[tuple[str, dict, bool]] = []
    while True:
        key = stdscr.getch()
        ch = chr(key) if 0 <= key < 256 else ""
        lower = ch.lower()
        if lower == "q" or key == 27:
            raise _Cancelled()
        if lower == "n":
            for kind, item in flagged:
                note = (
                    "mapping accepted as-is without correction" if item.get("_needs_verification")
                    else "accepted at low confidence without correction"
                )
                _append_note(item, note)
                item.pop("_needs_verification", None)
                item.pop("_edited", None)
            return True
        if lower == "a" and accepted:
            queue = [(k, it, True) for k, it in flagged] + [(k, it, False) for k, it in accepted]
            break
        if lower == "f" or key in ENTER_KEYS:
            if not flagged:
                # Nothing was flagged, and the reviewer didn't ask ("a")
                # to look at the auto-accepted ones either -- same
                # outcome as "no", just with nothing that needs a note.
                return True
            queue = [(k, it, True) for k, it in flagged]
            break

    for index, (kind, item, was_flagged) in enumerate(queue):
        _review_one_item(stdscr, kind, item, index, len(queue), threshold, was_flagged)
    return True


def _review_one_item(
    stdscr, kind: str, item: dict, index: int, total: int, threshold: float, was_flagged: bool = True,
) -> None:
    fields = PARAM_FIELDS if kind == "parameter" else METRIC_FIELDS
    action_cursor = 0
    while True:
        # Drawing the item panel is its own function, called both once up
        # front (to compute where the menu starts) and again on every one
        # of _select_menu's own redraw frames below (via `header=`) --
        # otherwise navigating the action menu, or a terminal resize while
        # it's open, would erase this panel or leave stale text from a
        # previous width's layout showing through underneath it.
        def draw_panel() -> int:
            max_y, max_x = stdscr.getmaxyx()
            box_w = min(max_x - 2, 60)

            title_line = f"Review {index + 1}/{total}"
            if not was_flagged:
                title_line += "  (auto-accepted -- reviewing by request)"
            _safe_addstr(stdscr, 0, 0, title_line, curses.A_BOLD)
            _safe_addstr(stdscr, 1, 0, "─" * min(box_w, max_x - 1))

            _safe_addstr(stdscr, 3, 2, _label(item, kind), curses.A_BOLD)
            rows = [
                ("Name", item.get("semantic_name", "")),
                ("Type", _pretty(item.get("datatype"), "schema:")),
                ("Unit", _pretty(item.get("unit"), "unit:")),
                ("Quantity", (item.get("quantityKind") or "").rsplit("/", 1)[-1]),
                ("Confidence", f"{_confidence(item) * 100:.0f}%"),
            ]
            y = 5
            for label, value in rows:
                _safe_addstr(stdscr, y, 4, f"{label:<10} {value}")
                y += 1

            y += 1
            _safe_addstr(stdscr, y, 4, "Explanation", curses.A_UNDERLINE)
            y += 1
            explanation = (item.get("explanation") or "").strip()
            for i in range(0, max(1, len(explanation)), box_w - 4):
                if y >= max_y - len(_ACTIONS) - 3:
                    break
                _safe_addstr(stdscr, y, 4, explanation[i:i + box_w - 4])
                y += 1
            return y

        y = draw_panel()
        stdscr.refresh()
        max_y, max_x = stdscr.getmaxyx()

        menu_y = y + 2
        choice = _select_menu(stdscr, menu_y, "? Action", _ACTIONS, start=action_cursor, header=draw_panel)
        if choice is None:
            # Cancelled from inside a single item's menu -- treat as "skip
            # this item" rather than aborting the whole queue; a full
            # queue cancel is only triggered from the summary screen.
            return
        action_cursor = choice
        action = _ACTIONS[choice]

        if action in ("Accept", "Skip"):
            if item.get("_edited"):
                note = "manually corrected by reviewer"
            elif item.get("_needs_verification"):
                note = "mapping accepted as-is without correction"
            elif was_flagged:
                note = "accepted at low confidence without correction"
            else:
                # Wasn't flagged at all -- the reviewer specifically asked
                # ("A" on the summary screen) to look at the items that
                # were already auto-accepted, and chose to leave this one
                # as-is. Distinct wording from the low-confidence case
                # above: this item met the threshold fine, a human just
                # double-checked it on request.
                note = "reviewed and accepted (previously auto-accepted)"
            _append_note(item, note)
            item.pop("_needs_verification", None)
            item.pop("_edited", None)
            return

        field = _FIELD_BY_ACTION[action]
        if field == "datatype":
            dt_choice = _select_menu(
                stdscr, menu_y, "? New datatype", list(ALLOWED_DATATYPES), header=draw_panel,
            )
            if dt_choice is None:
                continue
            new_value = ALLOWED_DATATYPES[dt_choice]
        else:
            max_y, max_x = stdscr.getmaxyx()
            prompt_y = min(menu_y + len(_ACTIONS) + 2, max_y - 1)
            typed = _text_input(stdscr, prompt_y, f"New value for {field} (current: {item.get(field)!r}): ")
            if typed is None:
                continue
            new_value = typed

        applied, warning = _apply_field_edit(item, field, new_value)
        max_y, max_x = stdscr.getmaxyx()
        confirm_y = min(menu_y + len(_ACTIONS) + 3, max_y - 1)
        msg = f"✓ {field} -> {applied!r}"
        if warning:
            msg += f"  (warning: {warning})"
        _safe_addstr(stdscr, confirm_y, 0, msg)
        stdscr.refresh()
        # Flash the confirmation briefly and move straight on -- NOT a
        # second stdscr.getch() wait. That used to require an extra,
        # easy-to-miss "press any key" dismiss keystroke before the next
        # real action could be chosen; anyone who didn't realize that was
        # needed would press what they actually wanted next (e.g. Accept,
        # or the digit for another action) and have it silently eaten by
        # this screen instead -- reading as "the menu doesn't respond to
        # toggling/editing" even though the edit itself had gone through.
        # The edited value is already visible above in the item's own
        # detail rows (they're re-read from `item` on every redraw), so
        # nothing about the edit is actually lost by not blocking here.
        curses.napms(500)


# =============================================================================
# Outputs step
# =============================================================================

def output_picker(current: dict[str, bool], filenames: dict[str, str]) -> dict[str, bool] | None:
    """Curses "which artifacts should this run produce" picker -- a preset
    menu (Standard / Metadata only / Reproducibility / Review only /
    Custom) shown first; choosing "Custom" opens a checkbox screen over
    the same three togglable outputs (dataset/snakefile/review -- the
    benchmark description itself is always generated, so it's shown as a
    locked line rather than a fourth checkbox).

    `current` seeds the Custom screen's initial checked state (e.g. a
    previously saved selection) and `filenames` maps the same three keys
    to the actual filename each one will produce, shown next to its
    purpose so a technical reviewer can see exactly what's about to be
    written.

    Returns the resolved {"dataset": bool, "snakefile": bool, "review":
    bool} dict, or None if curses isn't usable or the reviewer cancelled
    (the caller falls back to metadata.builder's plain-text equivalent).
    """
    if curses is None:
        return None
    try:
        return curses.wrapper(_output_picker_impl, current, filenames)
    except _Cancelled:
        return None
    except Exception:
        return None


def _output_picker_impl(stdscr, current: dict[str, bool], filenames: dict[str, str]) -> dict[str, bool]:
    curses.curs_set(0)
    stdscr.keypad(True)

    # Drawn via a function, not inline, so _select_menu_with_details can
    # call it again on every one of its own redraw frames (see its
    # docstring) -- this keeps the artifact list visible above the preset
    # menu on every keystroke, including right after a terminal resize,
    # instead of it only being drawn once and then getting erased (or
    # ghosted over) the first time the menu redraws.
    def draw_header() -> int:
        _safe_addstr(stdscr, 0, 0, "benchmantic can produce the following artifacts:", curses.A_BOLD)
        y = 2
        _safe_addstr(stdscr, y, 0, "  Benchmark description  (always generated)")
        y += 1
        _safe_addstr(stdscr, y, 0, "  Review report          (always generated)")
        y += 1
        for key, (name, desc) in OUTPUT_ITEM_INFO.items():
            mark = "x" if current.get(key) else " "
            _safe_addstr(stdscr, y, 0, f"  [{mark}] {name}  ({filenames.get(key, '')})")
            y += 1
        return y

    y = draw_header()
    stdscr.refresh()

    preset_names = list(OUTPUT_PRESETS) + ["Custom"]
    preset_details = [OUTPUT_PRESET_INFO.get(name, "") for name in preset_names]
    choice = _select_menu_with_details(
        stdscr, y + 2, "? Choose outputs", preset_names, preset_details, header=draw_header,
    )
    if choice is None:
        raise _Cancelled()
    if preset_names[choice] != "Custom":
        return dict(OUTPUT_PRESETS[preset_names[choice]])

    keys = list(OUTPUT_ITEM_INFO)
    labels = [f"{OUTPUT_ITEM_INFO[k][0]}  ({filenames.get(k, '')})" for k in keys]
    checked = {i for i, k in enumerate(keys) if current.get(k)}
    result = _checkbox_list_impl(
        stdscr, "Select outputs",
        "Benchmark description and review report are always generated -- these are the rest.",
        labels, checked,
    )
    return {k: (i in result) for i, k in enumerate(keys)}
