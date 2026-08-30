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
#: The full 2x2 grid over the two real Custom-reachable toggles (dataset,
#: snakefile) is covered exactly once each, always with description=True:
#: both on ("Standard"), snakefile only ("Workflow + Description"), dataset
#: only ("Description + Dataset"), neither ("Description only"). "Workflow
#: only" is a fifth, DELIBERATELY NOT Custom-reachable preset: it's the
#: only way to turn "description" off (which also forces "dataset" off --
#: the dataset sidecar's "schema:isPartOf" link points at the benchmark
#: file, which wouldn't exist -- and skips the Groq/OpenAI call and the
#: review report entirely, see metadata.builder's Infer & review step).
#: Keeping description=False out of the general Custom checkbox screen
#: avoids an incoherent combination (dataset=True, description=False)
#: that Custom's plain toggle-loop UI has no natural way to prevent a
#: reviewer from picking. Display names only -- the dict keys themselves
#: are purely internal to this interactive picker and its plain-text
#: fallback (metadata.builder._plain_outputs_prompt()); they're unrelated
#: to and don't need to match the `--outputs <preset>` CLI values
#: (metadata.builder._OUTPUT_PRESET_ALIASES), which are their own separate
#: kebab-case vocabulary ("standard", "snakefile-only", ...) kept stable
#: for scripts/CI regardless of how this menu's labels are worded.
OUTPUT_PRESETS: dict[str, dict[str, bool]] = {
    "Standard": {"description": True, "dataset": True, "snakefile": True},
    "Workflow + Description": {"description": True, "dataset": False, "snakefile": True},
    "Workflow only": {"description": False, "dataset": False, "snakefile": True},
    "Description only": {"description": True, "dataset": False, "snakefile": False},
    "Description + Dataset": {"description": True, "dataset": True, "snakefile": False},
}

#: One-line, plain-language description of what each preset actually
#: does, shown next to its name wherever a preset is offered (the curses
#: picker below and metadata.builder's plain-text fallback) instead of
#: just a comma list of included artifact keys -- kept deliberately short
#: (one line each) so the preset menu reads at a glance; the fuller
#: picture (no LLM/Groq call in "Workflow only", the unit-suffix caveat
#: that mode carries, exactly which files each preset skips) lives in
#: README.md/SKILL.md and in this screen's own live per-file preview
#: pane, not repeated here.
OUTPUT_PRESET_INFO: dict[str, str] = {
    "Standard": "Semantic description, dataset provenance, and Snakefile.",
    "Workflow + Description": "Semantic description and Snakefile -- no dataset provenance.",
    "Workflow only": "Reproducible Snakefile workflow from discovered parameter values.",
    "Description only": "Semantic description -- no dataset provenance, no Snakefile.",
    "Description + Dataset": "Semantic description and dataset provenance -- no Snakefile.",
    "Custom": "Choose dataset provenance and/or Snakefile individually.",
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

    Scrolls the option list (independently of whatever's above it via
    `header`) if there isn't room for all of it plus a footer row -- same
    reasoning as _select_menu_with_details' own scrolling.
    """
    cursor = max(0, min(start, len(options) - 1))
    top = 0
    while True:
        stdscr.erase()
        if header is not None:
            header()
        max_y, max_x = stdscr.getmaxyx()
        visible_rows = max(1, max_y - (y0 + 1) - 1)  # title row + 1 footer row
        if cursor < top:
            top = cursor
        if cursor >= top + visible_rows:
            top = cursor - visible_rows + 1
        top = max(0, min(top, max(0, len(options) - visible_rows)))
        _safe_addstr(stdscr, y0, 0, title, curses.A_BOLD)
        for row, i in enumerate(range(top, min(len(options), top + visible_rows))):
            attr = curses.A_REVERSE if i == cursor else 0
            marker = "❯ " if i == cursor else "  "
            digit = f"{i + 1}) " if i < 9 else "   "
            _safe_addstr(stdscr, y0 + 1 + row, 0, f"{marker}{digit}{options[i]}", attr)
        footer = "  ↑/↓ or a number to highlight, enter to confirm, q cancel"
        if top + visible_rows < len(options):
            footer += "  (more below)"
        elif top > 0:
            footer += "  (more above)"
        _safe_addstr(stdscr, min(y0 + 1 + visible_rows, max_y - 1), 0, footer)
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

    Scrolls when the options (each possibly several lines once its detail
    text wraps) don't all fit the terminal's current height -- a fixed
    terminal height plus six presets with multi-line descriptions meant
    the last couple of options could land entirely off-screen with no way
    to reach them short of resizing the terminal itself; up/down (and the
    digit jump) now scroll the visible window to keep whatever's
    highlighted in view, the same way _checkbox_list_impl already does
    for the parameter list.
    """
    cursor = max(0, min(start, len(options) - 1))
    top_line = 0
    while True:
        stdscr.erase()
        if header is not None:
            header()
        max_y, max_x = stdscr.getmaxyx()
        detail_width = max(10, max_x - 7)
        wrapped = [_wrap_text(d, detail_width) if d else [] for d in details]

        # Flatten into one line per row to draw -- an option's own header
        # line, then zero or more indented detail lines -- so scrolling
        # can work at row granularity regardless of how many lines any
        # one option's wrapped description takes up.
        lines: list[tuple[int, str, bool]] = []
        option_start: list[int] = []
        for i, opt in enumerate(options):
            option_start.append(len(lines))
            lines.append((i, opt, True))
            for dline in wrapped[i]:
                lines.append((i, dline, False))

        visible_rows = max(1, max_y - (y0 + 1) - 1)  # title row + 1 footer row
        cur_start = option_start[cursor]
        cur_span = 1 + len(wrapped[cursor])
        if cur_start < top_line:
            top_line = cur_start
        if cur_start + cur_span > top_line + visible_rows:
            # Scroll down just enough to show as much of this option as
            # fits -- but never past its OWN header line. Without this
            # cap, an option whose wrapped detail text alone is taller
            # than the whole visible window (cur_span > visible_rows,
            # possible for a long detail line on a short/narrow terminal)
            # would push top_line past cur_start, scrolling the
            # option's name itself off the top of the screen -- the exact
            # bug this scrolling was added to fix, just relocated. Losing
            # a few of its own detail lines off the bottom is an
            # acceptable tradeoff; losing the option's own name/highlight
            # off the top is not.
            top_line = min(cur_start + cur_span - visible_rows, cur_start)
        top_line = max(0, min(top_line, max(0, len(lines) - visible_rows)))

        _safe_addstr(stdscr, y0, 0, title, curses.A_BOLD)
        y = y0 + 1
        for li in range(top_line, min(len(lines), top_line + visible_rows)):
            idx, text, is_header_line = lines[li]
            if is_header_line:
                attr = curses.A_REVERSE if idx == cursor else 0
                marker = "❯ " if idx == cursor else "  "
                digit = f"{idx + 1}) " if idx < 9 else "   "
                _safe_addstr(stdscr, y, 0, f"{marker}{digit}{text}", attr)
            else:
                _safe_addstr(stdscr, y, 6, text)
            y += 1
        footer = "  ↑/↓ or a number to highlight, enter to confirm, q cancel"
        if top_line + visible_rows < len(lines):
            footer += "  (more below)"
        elif top_line > 0:
            footer += "  (more above)"
        _safe_addstr(stdscr, min(y0 + 1 + visible_rows, max_y - 1), 0, footer)
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
    """Text prompt drawn starting at row `y`, using a small hand-rolled
    line editor (not curses' echo+getstr, see below) -- keeps the whole
    interaction inside the same screen. Returns the edited text (stripped),
    or None if the reviewer cancelled with Escape (in which case the field
    is left completely untouched, same as before this was ever entered).

    `initial` pre-fills the editable buffer with the field's current value
    (the caller passes `item.get(field)`) -- editing now means what it
    normally means: the cursor starts at the end of the existing text,
    Left/Right (or Ctrl-A/Ctrl-E/Home/End) move within it, Backspace/Delete
    remove one character on either side of the cursor, and typing inserts
    at the cursor position rather than overwriting anything. Before this,
    the field started BLANK regardless of its current value -- the current
    value was only ever shown as read-only text in the prompt above, so
    changing one word deep inside a long explanation meant retyping the
    entire sentence from scratch. curses' own getstr() can't do in-place
    editing at all (no pre-fill, no cursor movement, Escape isn't even
    exposed to it), which is why this reimplements a minimal line editor
    with getch() instead -- it's more code, but it's what "edit" actually
    means here.

    If the buffer is wider than the available line width (routine for a
    100+ character explanation), the visible window scrolls horizontally
    to keep the cursor in view, the same idea as the digit-jump buffer in
    _checkbox_list_impl. Only single-byte/ASCII characters can be newly
    typed in -- a full UTF-8-aware line editor (multi-byte sequences
    arrive one getch() call at a time) is more than this needs; existing
    non-ASCII text in `initial` still displays and edits around fine,
    since it's already a decoded Python str, not something this function
    has to decode itself.

    `prompt` (e.g. "New value for explanation (current: '...a full
    sentence...'): ") is word-wrapped across as many lines as it needs to
    fit the window's actual width -- it used to be a single _safe_addstr
    call that silently CLIPPED anything past the window edge, which for a
    field with a long current value meant most of it was simply invisible
    on anything short of a maximized terminal. Typing always happens on
    its own line, right after however many lines the prompt took,
    starting at a fixed column -- decoupling the input line from the
    prompt's raw length (its *line count* now, not character count) is
    what keeps a long current value from ever pinning the cursor
    off-screen with no room left to edit into.
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

    buffer = list(initial)
    cursor = len(buffer)
    view_offset = 0
    stdscr.keypad(True)
    curses.curs_set(1)
    try:
        while True:
            max_y, max_x = stdscr.getmaxyx()
            avail = max(1, max_x - 3)
            if cursor - view_offset >= avail:
                view_offset = cursor - avail + 1
            if cursor < view_offset:
                view_offset = cursor
            view_offset = max(0, view_offset)
            text = "".join(buffer)
            _safe_addstr(stdscr, input_y, 0, "> ")
            stdscr.clrtoeol()
            _safe_addstr(stdscr, input_y, 2, text[view_offset:view_offset + avail])
            with contextlib.suppress(curses.error):
                stdscr.move(input_y, 2 + (cursor - view_offset))
            stdscr.refresh()

            key = stdscr.getch()
            if key in ENTER_KEYS:
                return "".join(buffer).strip() or None
            if key == 27:  # Escape -- cancel, field left untouched
                return None
            if key in (curses.KEY_LEFT,):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_RIGHT,):
                cursor = min(len(buffer), cursor + 1)
            elif key in (curses.KEY_HOME, 1):  # 1 = Ctrl-A
                cursor = 0
            elif key in (curses.KEY_END, 5):  # 5 = Ctrl-E
                cursor = len(buffer)
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                if cursor > 0:
                    del buffer[cursor - 1]
                    cursor -= 1
            elif key in (curses.KEY_DC,):  # Delete (forward)
                if cursor < len(buffer):
                    del buffer[cursor]
            elif 32 <= key < 127:  # printable ASCII -- insert at cursor
                buffer.insert(cursor, chr(key))
                cursor += 1
            # Anything else (an unrecognized/non-ASCII byte, a function
            # key with no binding here) is silently ignored rather than
            # inserted as garbage or crashing the editor.
    finally:
        curses.curs_set(0)


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
    colors_ok = _init_colors()
    max_y, max_x = stdscr.getmaxyx()
    # A long list (the 38-parameter selector is the motivating case) reads
    # much better split into two side-by-side columns -- numbered 1..split
    # on the left, split+1..N on the right -- than as one long scroll a
    # reviewer has to page all the way down just to see the last few
    # items. Only worth it once there's enough width for two real columns
    # AND enough items that a single column would need real scrolling
    # anyway; a short list (e.g. the Outputs step's own 2-item Custom
    # toggle) stays single-column, where a second, mostly-empty column
    # would just look broken.
    two_col = len(labels) > 12 and max_x >= 100
    if two_col:
        return _checkbox_list_two_col(stdscr, title, subtitle, labels, checked, colors_ok)
    return _checkbox_list_one_col(stdscr, title, subtitle, labels, checked, colors_ok)


def _checkbox_list_one_col(stdscr, title: str, subtitle: str, labels: list[str], checked: set[int], colors_ok: bool) -> set[int]:
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


def _checkbox_list_two_col(stdscr, title: str, subtitle: str, labels: list[str], checked: set[int], colors_ok: bool) -> set[int]:
    """Side-by-side two-column variant of the checkbox list, for a long
    list on a wide terminal (the 38-parameter selector is the motivating
    case): items 1..split numbered down the left column, split+1..N down
    the right, with Tab hopping the highlight to the matching row in the
    other column -- reading down one short column, then the other, beats
    scrolling through one long one.

    Same underlying selection semantics as the single-column version
    (Space toggles, a digit typed anywhere jumps to that absolute item
    number in whichever column it lands in, Enter with no digits pending
    confirms, 'a'/'n' select all/none, 'q'/Escape cancels) -- only the
    layout and the extra Tab binding differ.
    """
    n = len(labels)
    split = -(-n // 2)  # ceil(n / 2): left column gets the extra item if n is odd
    cursor = 0
    top_row = 0  # shared vertical scroll offset, in rows-within-a-column
    buffer = ""

    def col_bounds(i: int) -> tuple[int, int]:
        return (0, split - 1) if i < split else (split, n - 1)

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        digit_width = len(str(n))
        col_w = max(24, (max_x - 6) // 2)

        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD | _cp(colors_ok, 1))
        selected_badge = f"Selected  {len(checked)}/{n}"
        _safe_addstr(stdscr, 0, max(0, max_x - len(selected_badge) - 1), selected_badge, _cp(colors_ok, 2) | curses.A_BOLD)
        for line in _wrap_text(subtitle, max(10, max_x - 1)):
            _safe_addstr(stdscr, 1, 0, line)
            break  # keep the header to one line; the rest is still in subtitle for callers that need it

        info_y = 2
        _safe_addstr(stdscr, info_y, 0, "Use ↑/↓ to navigate, SPACE to toggle, TAB to switch columns, ENTER when done.")

        legend_y = info_y + 2
        _safe_addstr(stdscr, legend_y, 0, "Legend: [x] selected   [ ] not selected")
        commands = "a: all   n: none   q: cancel"
        _safe_addstr(stdscr, legend_y, max(0, max_x - len(commands) - 1), commands)

        header_y = legend_y + 2
        left_x, right_x = 0, col_w + 3
        _safe_addstr(stdscr, header_y, left_x, "#".rjust(digit_width) + "  Parameter", curses.A_UNDERLINE)
        _safe_addstr(stdscr, header_y, right_x, "#".rjust(digit_width) + "  Parameter", curses.A_UNDERLINE)

        list_top = header_y + 1
        footer_selected_lines = _wrap_text(
            f"{len(checked)} parameter(s) selected: " + ", ".join(labels[i] for i in sorted(checked)) if checked
            else "No parameters selected.",
            max(10, max_x - 1),
        )[:3]
        bottom_bar = "↑/↓ navigate   SPACE toggle   TAB switch column   a all   n none   ENTER finish   q cancel"
        # rows reserved below the list: 1 blank + up to 3 summary lines + 1 blank + bottom bar
        reserved_bottom = 1 + len(footer_selected_lines) + 1 + 1
        visible_rows = max(1, max_y - list_top - reserved_bottom)

        lo, hi = col_bounds(cursor)
        rel = cursor - lo
        if rel < top_row:
            top_row = rel
        if rel >= top_row + visible_rows:
            top_row = rel - visible_rows + 1
        top_row = max(0, top_row)

        for row in range(visible_rows):
            for col, (lo_c, hi_c, x) in enumerate([(0, split - 1, left_x), (split, n - 1, right_x)]):
                i = lo_c + top_row + row
                if i > hi_c:
                    continue
                mark = "x" if i in checked else " "
                attr = _cp(colors_ok, 4) | curses.A_BOLD if i == cursor else 0
                marker = "❯ " if i == cursor else "  "
                num = f"{i + 1:>{digit_width}}"
                label = labels[i][:max(4, col_w - digit_width - 14)]
                line = f"{marker}{num}  {label}"
                _safe_addstr(stdscr, list_top + row, x, line.ljust(col_w), attr)
                mark_attr = _cp(colors_ok, 2) if mark == "x" else 0
                _safe_addstr(stdscr, list_top + row, x + col_w - 4, f"[{mark}]", mark_attr | (attr if i == cursor else 0))

        y = list_top + visible_rows + 1
        summary_attr = _cp(colors_ok, 2) if checked else 0
        for line in footer_selected_lines:
            _safe_addstr(stdscr, y, 0, line, summary_attr)
            y += 1
        y += 1
        _safe_addstr(stdscr, min(y, max_y - 1), 0, bottom_bar[:max(0, max_x - 1)])

        if buffer:
            goto_y = min(y, max_y - 1)
            goto_text = f"Go to #: {buffer}"
            _safe_addstr(stdscr, goto_y, 0, goto_text, curses.A_BOLD)
            curses.curs_set(1)
            with contextlib.suppress(curses.error):
                stdscr.move(goto_y, len(goto_text))
        else:
            curses.curs_set(0)
        stdscr.refresh()

        key = stdscr.getch()
        if ord("0") <= key <= ord("9"):
            if len(buffer) < digit_width:
                buffer += chr(key)
                idx = int(buffer) - 1
                if 0 <= idx < n:
                    cursor = idx
            continue
        if key in (curses.KEY_BACKSPACE, 127, 8):
            buffer = buffer[:-1]
            if buffer:
                idx = int(buffer) - 1
                if 0 <= idx < n:
                    cursor = idx
            continue
        if key in ENTER_KEYS:
            if buffer:
                idx = int(buffer) - 1
                buffer = ""
                if 0 <= idx < n:
                    cursor = idx
                    checked.symmetric_difference_update({idx})
                continue
            return checked
        if key in (27, ord("q"), ord("Q")):
            if buffer:
                buffer = ""
                continue
            raise _Cancelled()
        buffer = ""
        if key == ord("\t"):
            other_lo, other_hi = col_bounds(split if cursor < split else 0)
            cursor = min(other_lo + rel, other_hi)
        elif key in (curses.KEY_UP, ord("k")):
            lo, hi = col_bounds(cursor)
            cursor = lo + (cursor - lo - 1) % (hi - lo + 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            lo, hi = col_bounds(cursor)
            cursor = lo + (cursor - lo + 1) % (hi - lo + 1)
        elif key in (curses.KEY_LEFT,) and cursor >= split:
            cursor -= split
        elif key in (curses.KEY_RIGHT,) and cursor < split:
            cursor = min(cursor + split, n - 1)
        elif key == ord(" "):
            checked.symmetric_difference_update({cursor})
        elif key in (ord("a"), ord("A")):
            checked = set(range(n))
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
            current_value = item.get(field) or ""
            typed = _text_input(
                stdscr, prompt_y,
                f"Edit {field} (←/→ move, Backspace/Delete remove, Enter save, Esc cancel):",
                initial=str(current_value),
            )
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
        # After a successful edit, land back on "Accept" rather than
        # staying on the action just used -- the common flow is "fix the
        # one thing that was wrong, then accept and move to the next
        # item", so that's the more useful default to highlight next,
        # not a re-run of the same edit. (A CANCELLED edit -- the `continue`
        # branches above, for `dt_choice is None`/`typed is None` -- does
        # NOT reach this line, so backing out of a submenu/text prompt
        # still leaves the highlight where it was, in case the intent was
        # just to retry the same action.)
        action_cursor = 0


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


def _init_colors() -> bool:
    """Turn color on if this terminal supports it. Everything colored
    below is drawn through _cp(), which falls back to a plain attribute
    (or nothing) when this returns False -- color is a nice-to-have layered
    on top of a screen that already works without it (a monochrome
    terminal, or one where curses.start_color() itself fails), never a
    requirement for the Outputs picker to function."""
    if not curses.has_colors():
        return False
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)                    # section headers
        curses.init_pair(2, curses.COLOR_GREEN, -1)                   # included artifact
        curses.init_pair(3, curses.COLOR_YELLOW, -1)                  # type badges / "Recommended"
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)    # highlighted preset row
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_GREEN)   # preview summary banner
        curses.init_pair(6, curses.COLOR_RED, -1)                     # skipped artifact
        return True
    except curses.error:
        return False


def _cp(colors_ok: bool, n: int, fallback: int = 0) -> int:
    return curses.color_pair(n) if colors_ok else fallback


def _artifact_cards(filenames: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """(key, title, type badge, one-line description) for all four
    artifacts this tool can produce, in the fixed display order used
    throughout the Outputs step -- the two always-on ones (benchmark
    description, review report) alongside the two real OUTPUT_ITEM_INFO
    toggles, so the info panel always shows the complete picture regardless
    of what's currently selected."""
    return [
        ("description", "Benchmark description", "JSON-LD",
         "Machine-readable semantic description of the benchmark and execution metadata."),
        ("dataset", OUTPUT_ITEM_INFO["dataset"][0], "JSON-LD", OUTPUT_ITEM_INFO["dataset"][1]),
        ("snakefile", OUTPUT_ITEM_INFO["snakefile"][0], "Snakefile", OUTPUT_ITEM_INFO["snakefile"][1]),
        ("review", "Review report", "Markdown",
         "Human-readable summary of semantic inference and review decisions."),
    ]


def _draw_cards(stdscr, y0: int, cards: list[tuple[str, str, str, str]], colors_ok: bool) -> int:
    """Draws the "what will be produced" info panel: an intro line, then
    the artifact cards laid out in as many columns as the terminal is wide
    enough for (4 down to 1 as the terminal narrows), each showing its
    name, file-type badge, and a short description. Returns the row just
    below the panel, for whatever's drawn next."""
    _safe_addstr(stdscr, y0, 0, "What will be produced?", curses.A_BOLD | _cp(colors_ok, 1))
    y0 += 1
    _safe_addstr(
        stdscr, y0, 0,
        "benchmantic analyzes your simulation benchmark and can generate these artifacts:",
    )
    y0 += 2

    max_y, max_x = stdscr.getmaxyx()
    min_card_w = 26
    cols = max(1, min(len(cards), (max_x + 2) // (min_card_w + 2)))
    card_w = max(min_card_w, (max_x - (cols - 1) * 2) // cols)

    y = y0
    for row_start in range(0, len(cards), cols):
        row = cards[row_start:row_start + cols]
        wrapped = [_wrap_text(desc, max(10, card_w - 1))[:2] for _, _, _, desc in row]
        body_h = max((len(w) for w in wrapped), default=1)
        for i, (_key, title, badge, _desc) in enumerate(row):
            x = i * (card_w + 2)
            if y >= max_y or x >= max_x:
                continue
            _safe_addstr(stdscr, y, x, title[:card_w], curses.A_BOLD)
            _safe_addstr(stdscr, y + 1, x, f"[{badge}]", _cp(colors_ok, 3))
            for li, line in enumerate(wrapped[i]):
                _safe_addstr(stdscr, y + 2 + li, x, line)
        y += 2 + body_h + 1
    return y


def _draw_section_rule(stdscr, y: int, label: str, note: str) -> None:
    """A single "- Label ------------------- right-aligned note" divider
    line, filling the space between with dashes -- marks the start of the
    "Choose outputs" section the same way a rule under a heading would."""
    max_y, max_x = stdscr.getmaxyx()
    prefix = f"- {label} "
    _safe_addstr(stdscr, y, 0, prefix, curses.A_BOLD)
    note_x = max(len(prefix), max_x - len(note) - 1)
    if note_x > len(prefix):
        _safe_addstr(stdscr, y, len(prefix), "-" * (note_x - len(prefix)))
    _safe_addstr(stdscr, y, note_x, note)


def _output_two_pane(
    stdscr, y0: int, preset_names: list[str], filenames: dict[str, str], colors_ok: bool, header,
) -> int | None:
    """The live two-pane Outputs picker: a preset list on the left, and a
    file-by-file preview of whatever preset is currently highlighted on
    the right, updating with every arrow keypress -- so what Enter is
    about to confirm is always visible before it's pressed, rather than a
    separate plain-text "Generate these files? [Y/n]" prompt after the
    fact (metadata.builder skips that follow-up prompt entirely once this
    screen has returned a selection, for exactly that reason).

    Same controls as every other menu in this module: ↑/↓ (or j/k) or a
    digit 1-9 moves the highlight, Enter confirms whatever's highlighted,
    q/Escape cancels.
    """
    cursor = 0
    top = 0
    preview_rows = [
        ("description", "Benchmark description"),
        ("dataset", "Dataset description"),
        ("snakefile", "Reproducible workflow"),
        ("review", "Review report"),
    ]
    while True:
        stdscr.erase()
        header()
        max_y, max_x = stdscr.getmaxyx()
        left_w = max(28, min(48, (max_x - 3) * 2 // 5))
        right_x = left_w + 3
        right_w = max(20, max_x - right_x)

        rows_per_preset = 2
        avail_rows = max(rows_per_preset, max_y - y0 - 3)
        visible_n = max(1, avail_rows // rows_per_preset)
        if cursor < top:
            top = cursor
        if cursor >= top + visible_n:
            top = cursor - visible_n + 1
        top = max(0, min(top, max(0, len(preset_names) - visible_n)))

        # ---- left: preset list ----
        y = y0
        for i in range(top, min(len(preset_names), top + visible_n)):
            name = preset_names[i]
            selected = i == cursor
            label = f"{'> ' if selected else '  '}{i + 1}) {name}"
            if name == "Standard":
                label += "  [Recommended]"
            attr = (_cp(colors_ok, 4) | curses.A_BOLD) if selected else 0
            _safe_addstr(stdscr, y, 0, label[:left_w].ljust(left_w) if selected else label[:left_w], attr)
            y += 1
            desc = OUTPUT_PRESET_INFO.get(name, "")
            _safe_addstr(stdscr, y, 4, desc[:max(0, left_w - 4)])
            y += 1
        if top + visible_n < len(preset_names):
            _safe_addstr(stdscr, y, 0, "(more below)", curses.A_DIM)
        elif top > 0:
            _safe_addstr(stdscr, y0 - 1, left_w + 3, "")  # no-op, kept for symmetry

        # ---- right: live preview of the highlighted preset ----
        preset_name = preset_names[cursor]
        selection = OUTPUT_PRESETS.get(preset_name)  # None for "Custom"
        ry = y0
        _safe_addstr(stdscr, ry, right_x, f"Preset preview: {preset_name}"[:right_w], curses.A_BOLD)
        ry += 2
        if selection is None:
            for line in _wrap_text(OUTPUT_PRESET_INFO.get("Custom", ""), max(10, right_w - 1)):
                _safe_addstr(stdscr, ry, right_x, line)
                ry += 1
        else:
            _safe_addstr(stdscr, ry, right_x, "Artifact".ljust(24) + "Filename", curses.A_UNDERLINE)
            ry += 1
            for key, label in preview_rows:
                included = selection.get("description", True) if key == "review" else selection.get(key, True)
                mark = "✓" if included else "✗"
                fname = filenames.get(key, "") if included else "(skipped)"
                _safe_addstr(stdscr, ry, right_x, mark, _cp(colors_ok, 2 if included else 6, curses.A_BOLD))
                _safe_addstr(stdscr, ry, right_x + 2, (label.ljust(24) + fname)[:max(0, right_w - 2)])
                ry += 1
            ry += 1
            summary_attr = _cp(colors_ok, 5)
            for line in _wrap_text(OUTPUT_PRESET_INFO.get(preset_name, ""), max(10, right_w - 2)):
                _safe_addstr(stdscr, ry, right_x, f" {line}".ljust(right_w), summary_attr)
                ry += 1

        # ---- footer ----
        footer_y = max_y - 2
        _safe_addstr(stdscr, footer_y, 0, "-" * max(0, max_x - 1))
        left_help = "↑/↓ Navigate   Enter to confirm   q to cancel"
        right_help = "Need help? Run benchmantic --help"
        _safe_addstr(stdscr, footer_y + 1, 0, left_help)
        _safe_addstr(stdscr, footer_y + 1, max(0, max_x - len(right_help) - 1), right_help)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(preset_names)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(preset_names)
        elif ord("1") <= key <= ord("9") and (key - ord("1")) < len(preset_names):
            cursor = key - ord("1")
        elif key in ENTER_KEYS:
            return cursor
        elif key in (27, ord("q"), ord("Q")):
            return None


def _output_picker_impl(stdscr, current: dict[str, bool], filenames: dict[str, str]) -> dict[str, bool]:
    curses.curs_set(0)
    stdscr.keypad(True)
    colors_ok = _init_colors()
    cards = _artifact_cards(filenames)

    # Drawn via a function, not inline, so both the two-pane layout and
    # _select_menu_with_details' own fallback can call it again on every
    # redraw frame -- this keeps the artifact-card panel visible on every
    # keystroke, including right after a terminal resize, instead of it
    # only being drawn once and then getting erased (or ghosted over) the
    # first time the menu below it redraws.
    def draw_header() -> int:
        y = _draw_cards(stdscr, 0, cards, colors_ok)
        _draw_section_rule(stdscr, y, "Choose outputs", "Select one or more outputs to generate")
        return y + 1

    y = draw_header()
    stdscr.refresh()
    max_y, max_x = stdscr.getmaxyx()

    preset_names = list(OUTPUT_PRESETS) + ["Custom"]
    # The live two-pane layout (preset list + per-file preview side by
    # side, matching the mockup this screen is modeled on) needs real
    # width and height to not feel cramped -- on anything narrower/
    # shorter, fall back to the single-column stacked menu instead of
    # squeezing two columns into a terminal that can't comfortably fit
    # them; the same card header is still shown above either way.
    if max_x >= 92 and max_y - y >= 4 + 2 * len(preset_names):
        choice = _output_two_pane(stdscr, y, preset_names, filenames, colors_ok, draw_header)
    else:
        preset_details = [OUTPUT_PRESET_INFO.get(name, "") for name in preset_names]
        choice = _select_menu_with_details(
            stdscr, y, "? Choose outputs", preset_names, preset_details, header=draw_header,
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
    # Custom never turns "description" off -- see OUTPUT_PRESETS' comment
    # above for why that's only reachable via the "Workflow only" preset.
    selection = {k: (i in result) for i, k in enumerate(keys)}
    selection["description"] = True
    return selection
