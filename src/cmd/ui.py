# Copyright (c) 2026 Andrei Surugiu

"""Terminal rendering for the harness, on top of Rich.

The `show_*` helpers format model replies, tool results, and REPL messages.
The agent decides when to call them. The primitives below handle the drawing.

Paint text with a chainable style API. Every call returns a Rich `Text`, so
painted pieces nest, and join with `+`::

    ui.echo(ui.bg("purple", " agent "), "ready")
    ui.echo(ui.bold.fg("green", "✓"), ui.dim("saved in 1.2s"))
    badge = ui.bold.bg("purple")  # a style is a value: keep it, reuse it
    ui.echo(badge(" v2 "))

A color is a name from `PALETTE` ("accent", "purple", "muted", ...) or any
color that Rich knows ("magenta", "#ff8700", "color(93)"). A palette color has
a light shade for text and a deep shade for backgrounds. `bg` also picks black
or white text, whichever is easier to read on the background.

Show work in progress in a `Live` region at the bottom of the terminal.
Printed and streamed output moves up into the normal scrollback, and only the
unfinished tail and the status line repaint::

    with ui.Live() as live:
        live.status("Thinking")
        for chunk in chunks:
            text += chunk
            live.stream(ui.markdown(text))
        live.print(ui.dim("done"))
"""

from __future__ import annotations

import json
import readline  # ruff: ignore[unused-import] -- gives `input` line editing
import time
from typing import TYPE_CHECKING, ClassVar, Final, Self, final, overload, override

import rich.live
import rich.markdown
from rich.color import Color, blend_rgb
from rich.console import Console, Group
from rich.rule import Rule
from rich.segment import Segment, SegmentLines
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from pydantic_ai import ModelResponsePart, ToolCallPart
    from rich.console import (
        ConsoleOptions,
        JustifyMethod,
        RenderableType,
        RenderResult,
    )
    from rich.markdown import MarkdownElement

PALETTE: Final[dict[str, tuple[str, str]]] = {
    # Roles. Paint with these, and the look changes in one place.
    "accent": ("#a78bfa", "#7c3aed"),
    "muted": ("#a1a1aa", "#52525b"),
    "success": ("#4ade80", "#16a34a"),
    "warning": ("#fbbf24", "#d97706"),
    "error": ("#f87171", "#dc2626"),
    # Hues from the Tailwind palette: shade 400 for text, 600 for backgrounds.
    "red": ("#f87171", "#dc2626"),
    "orange": ("#fb923c", "#ea580c"),
    "yellow": ("#facc15", "#ca8a04"),
    "green": ("#4ade80", "#16a34a"),
    "cyan": ("#22d3ee", "#0891b2"),
    "blue": ("#60a5fa", "#2563eb"),
    "violet": ("#a78bfa", "#7c3aed"),
    "purple": ("#c084fc", "#9333ea"),
    "pink": ("#f472b6", "#db2777"),
    "gray": ("#a1a1aa", "#52525b"),
}
"""Named colors, as (shade for text, shade for backgrounds)."""

CODE_THEME: Final = "monokai"
"""The Pygments theme for code blocks in markdown."""

_PROMPT: Final = "❯ "  # ruff: ignore[ambiguous-unicode-character-string]
_PREVIEW_LINES: Final = 3  # tool output to show; the model gets all of it

_FRAMES: Final = "·✢✳✶✻✽✻✶✳✢"
_FRAME_SECONDS: Final = 0.12
_REPAINTS_PER_SECOND: Final = 12.5
_STREAM_SECONDS: Final = 0.05  # render a stream at most 20 times a second
_SHIMMER_SECONDS: Final = 1.6  # one sweep of the highlight across the status
_SHIMMER_WIDTH: Final = 3.0  # characters on each side of the highlight
_SHIMMER_GLOW: Final = "#f5f3ff"
_DETAIL_LINES: Final = 3  # status detail lines to show, from the end
_BRIGHT: Final = 128  # YIQ brightness above which black text reads better


def _color(name: str, *, background: bool) -> Color:
    text, back = PALETTE.get(name, (name, name))
    return Color.parse(back if background else text)


@final
class Paint:
    """A text style. Chain it to add more, call it with text to paint the text.

    Painting returns a Rich `Text`. The style applies under the styles that the
    parts already have, so the innermost style wins where two styles clash.
    """

    __slots__ = ("_style",)

    def __init__(self, style: Style | None = None) -> None:
        """Make a paint with a Rich `Style`. Most code uses the shortcuts."""
        self._style = style or Style()

    def __call__(self, *parts: str | Text) -> Text:
        """Paint text.

        Returns:
            The parts joined together, in this style.

        """
        text = Text.assemble(*parts)
        text.stylize_before(self._style)
        return text

    @override
    def __repr__(self) -> str:
        """Show the style, for debugging.

        Returns:
            For example `Paint('bold #ffffff on #9333ea')`.

        """
        return f"Paint({str(self._style)!r})"

    @property
    def bold(self) -> Paint:
        """Bold text."""
        return Paint(self._style + Style(bold=True))

    @property
    def dim(self) -> Paint:
        """Faint text, for things that matter less."""
        return Paint(self._style + Style(dim=True))

    @property
    def italic(self) -> Paint:
        """Italic text."""
        return Paint(self._style + Style(italic=True))

    @property
    def underline(self) -> Paint:
        """Underlined text."""
        return Paint(self._style + Style(underline=True))

    @overload
    def fg(self, color: str, /) -> Paint: ...
    @overload
    def fg(self, color: str, text: str | Text, /, *more: str | Text) -> Text: ...
    def fg(self, color: str, /, *text: str | Text) -> Paint | Text:
        """Color the text itself.

        Returns:
            The painted text if you give text, else a paint to use later.

        """
        return self._then(Style(color=_color(color, background=False)), text)

    @overload
    def bg(self, color: str, /) -> Paint: ...
    @overload
    def bg(self, color: str, text: str | Text, /, *more: str | Text) -> Text: ...
    def bg(self, color: str, /, *text: str | Text) -> Paint | Text:
        """Color the background, and the text too if it has no color yet.

        Returns:
            The painted text if you give text, else a paint to use later.

        """
        background = _color(color, background=True)
        if self._style.color is None:
            return self._then(
                Style(color=self._readable_on(background), bgcolor=background), text
            )
        return self._then(Style(bgcolor=background), text)

    def _then(self, style: Style, text: tuple[str | Text, ...]) -> Paint | Text:
        paint = Paint(self._style + style)
        return paint(*text) if text else paint

    @staticmethod
    def _readable_on(background: Color) -> Color:
        red, green, blue = background.get_truecolor()
        brightness = (299 * red + 587 * green + 114 * blue) / 1000
        return Color.parse("#000000" if brightness > _BRIGHT else "#ffffff")


_PLAIN: Final = Paint()
bold: Final = _PLAIN.bold
dim: Final = _PLAIN.dim
italic: Final = _PLAIN.italic
underline: Final = _PLAIN.underline
fg: Final = _PLAIN.fg
bg: Final = _PLAIN.bg

_ACCENT, _ACCENT_DEEP = PALETTE["accent"]
_MUTED, _MUTED_DEEP = PALETTE["muted"]
_LINK = PALETTE["blue"][0]

console: Final = Console(
    highlight=False,
    markup=False,
    emoji=False,
    theme=Theme(
        {
            "markdown.h1": f"bold {_ACCENT}",
            "markdown.h2": f"bold {_ACCENT}",
            "markdown.h3": "bold",
            "markdown.h4": "bold italic",
            "markdown.h5": "italic",
            "markdown.h6": "italic",
            "markdown.code": _ACCENT,
            "markdown.block_quote": f"italic {_MUTED}",
            "markdown.item.bullet": _ACCENT,
            "markdown.item.number": _ACCENT,
            "markdown.link": _LINK,
            "markdown.link_url": f"underline {_LINK}",
            "markdown.hr": _MUTED_DEEP,
            "markdown.table.border": _MUTED_DEEP,
            "markdown.table.header": "bold",
        }
    ),
)
"""The terminal. Printing goes through it, so that it knows about `Live`."""


def echo(*objects: RenderableType) -> None:
    """Print painted text, markdown, or any Rich renderable.

    Plain strings print as they are: Rich markup in them is not parsed.
    """
    console.print(*objects)


def ask(prompt: str = _PROMPT) -> str | None:
    """Read a line from the user, with line editing and history.

    The prompt is plain text. macOS Python uses libedit for line editing, and
    libedit either drops the color codes in a prompt or miscounts its width.

    Returns:
        What the user typed, or `None` when they press ctrl+d or ctrl+c.

    """
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None


@final
class _Heading(rich.markdown.Heading):
    LEVEL_ALIGN: ClassVar[dict[str, JustifyMethod]] = {
        f"h{level}": "left" for level in range(1, 7)
    }


@final
class _Rule(rich.markdown.HorizontalRule):
    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(style=console.get_style("markdown.hr", default="none"))
        yield Text()


@final
class _Markdown(rich.markdown.Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **rich.markdown.Markdown.elements,
        "heading_open": _Heading,
        "hr": _Rule,
    }


def markdown(text: str) -> RenderableType:
    """Format markdown, with headings on the left and highlighted code.

    Returns:
        A renderable for `echo`, `Live.print`, or `Live.stream`.

    """
    return _Markdown(text, code_theme=CODE_THEME)


@final
class _Bullet:
    def __init__(self, marker: Text, content: RenderableType) -> None:
        self._marker = marker
        self._content = content

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        indent = self._marker.cell_len + 1
        width = max(options.max_width - indent, 1)
        lines = console.render_lines(
            self._content, options.update_width(width), pad=False
        )
        while lines and not Segment.get_line_length(lines[0]):
            del lines[0]  # Rich starts a leading list with an empty line
        marker = [*self._marker.render(console), Segment(" ")]
        blank = [Segment(" " * indent)]
        for index, line in enumerate(lines):
            yield from blank if index else marker
            yield from line
            yield Segment.line()


def bullet(marker: str | Text, content: RenderableType) -> RenderableType:
    """Hang a marker in front of content, like an item in a list.

    Returns:
        The marker and a space, then the content. Lines after the first are
        indented to line up with the first.

    """
    return _Bullet(Text(marker) if isinstance(marker, str) else marker, content)


@final
class _Status:
    def __init__(self, hint: str) -> None:
        self.hint = hint
        self.label = "Working"
        self.detail: RenderableType | None = None
        self.since = time.monotonic()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        now = time.monotonic()
        frame = _FRAMES[int(now / _FRAME_SECONDS) % len(_FRAMES)]
        facts = [f"{now - self.since:.0f}s", *([self.hint] if self.hint else [])]
        yield Text.assemble(
            fg("accent", frame),
            " ",
            self._shimmer(f"{self.label}…", now),
            dim(f" {' · '.join(facts)}"),
        )
        if self.detail is None:
            return
        gutter = [*dim("  ┊ ").render(console)]
        width = max(options.max_width - 4, 1)
        lines = console.render_lines(
            self.detail, options.update_width(width), pad=False
        )
        for line in lines[-_DETAIL_LINES:]:
            yield from gutter
            yield from line
            yield Segment.line()

    @staticmethod
    def _shimmer(text: str, now: float) -> Text:
        base = _color("accent", background=False).get_truecolor()
        glow = Color.parse(_SHIMMER_GLOW).get_truecolor()
        sweep = len(text) + 2 * _SHIMMER_WIDTH
        center = (now % _SHIMMER_SECONDS) / _SHIMMER_SECONDS * sweep - _SHIMMER_WIDTH
        shimmer = Text()
        for index, char in enumerate(text):
            closeness = max(0.0, 1 - abs(index - center) / _SHIMMER_WIDTH)
            color = Color.from_triplet(blend_rgb(base, glow, closeness))
            shimmer.append(char, Style(color=color))
        return shimmer


@final
class Live:
    """A live region at the bottom of the terminal, with a spinning status line.

    Use it as a context manager. While it is open, print with `print` and
    `stream`, not `echo`. Each thing printed or streamed is a block, with a
    blank line before it. When the region closes, the status goes away and
    everything that was printed or streamed stays.
    """

    def __init__(self, *, hint: str = "", tail: int = 6, gap: int = 1) -> None:
        """Make a live region. It shows nothing until the `with` block starts.

        Args:
            hint: A short tip to show after the status, like "ctrl+c to stop".
            tail: How many of the newest streamed lines can still change.
            gap: How many blank lines go before each block.

        """
        self._status = _Status(hint)
        self._tail = tail
        self._gap = gap
        self._stream: RenderableType | None = None
        self._streamed = 0  # lines of the stream that are already in scrollback
        self._lines: list[list[Segment]] = []  # the rest, which can still change
        self._rendered_at = 0.0
        self._live = rich.live.Live(
            console=console,
            transient=True,
            refresh_per_second=_REPAINTS_PER_SECOND,
            get_renderable=self._render,
        )

    def __enter__(self) -> Self:
        """Start showing the status.

        Returns:
            The live region.

        """
        self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Move what is left of a stream to scrollback, and clear the status."""
        self._end_stream()
        self._live.stop()

    @property
    def elapsed(self) -> float:
        """Seconds since the status label changed."""
        return time.monotonic() - self._status.since

    def status(self, label: str, detail: RenderableType | None = None) -> None:
        """Say what is happening now, next to the spinner.

        Args:
            label: A word or two, like "Thinking". The time shown next to it
                starts again when the label changes.
            detail: Something to show below the label, like the latest
                thoughts. Only its last few lines show, and only until the
                next call.

        """
        if label != self._status.label:
            self._status.label = label
            self._status.since = time.monotonic()
        self._status.detail = detail

    def print(self, *objects: RenderableType) -> None:
        """Print a block above the live region, into scrollback."""
        self._end_stream()
        self._space()
        console.print(*objects)

    def stream(self, renderable: RenderableType, *, final: bool = False) -> None:
        """Show a block that is still growing, like text that arrives in chunks.

        Call it again with each bigger version. Lines that are more than `tail`
        lines above the bottom can't change anymore, so they move into
        scrollback, and only the lines below them repaint.

        Args:
            renderable: The block so far.
            final: Whether this is the whole block. Then all of it moves into
                scrollback, and the next `stream` starts a new block.

        """
        if self._stream is None:
            self._space()
        self._stream = renderable
        now = time.monotonic()
        if not final and now - self._rendered_at < _STREAM_SECONDS:
            return
        self._rendered_at = now
        lines = console.render_lines(renderable, pad=False)
        done = len(lines) if final else max(len(lines) - self._tail, self._streamed)
        with console:  # write scrollback and repaint at once, so nothing flickers
            if done > self._streamed:
                console.print(
                    SegmentLines(lines[self._streamed : done], new_lines=True)
                )
            self._streamed = done
            self._lines = lines[done:]
            self._live.refresh()
        if final:
            self._stream = None
            self._streamed = 0

    def _end_stream(self) -> None:
        if self._stream is not None:
            self.stream(self._stream, final=True)

    def _space(self) -> None:
        for _ in range(self._gap):
            console.print()

    def _render(self) -> RenderableType:
        gap: list[list[Segment]] = [[] for _ in range(self._gap)]
        return Group(SegmentLines(self._lines + gap, new_lines=True), self._status)


def show_model_info(name: str, url: str) -> None:
    """Show the harness banner, model name, and server URL."""
    echo(
        bold.bg("accent", " ✻ minimal-agentic-harness "),
        dim(f"{name} at {url}"),
    )


def show_tools_info(names: Iterable[str]) -> None:
    """Show the available tools and REPL command hints."""
    echo(dim(f"tools: {', '.join(names)} · /clear to start over · ctrl+d to quit"))
    echo()


def show_cleared() -> None:
    """Confirm that the conversation was cleared."""
    echo(dim("  ⎿ Cleared the conversation"))


def show_error(message: str) -> None:
    """Show the reason a turn failed."""
    echo(fg("error", f"  ⎿ {message}"))


def show_turn_summary(seconds: float, tokens: int) -> None:
    """Show the turn duration, output tokens, and tokens per second."""
    speed = f"{tokens / seconds:.0f} tok/s"
    echo()
    echo(dim(f"✻ Worked for {seconds:.1f}s · {tokens} tokens · {speed}"))


def show_response_part(
    part: ModelResponsePart, live: Live, *, done: bool = False
) -> None:
    """Show part of a reply. Thoughts pass by under the status, text stays."""
    match part.part_kind:
        case "thinking" if done:
            live.print(dim(f"✻ Thought for {live.elapsed:.1f}s"))
        case "thinking":
            live.status("Thinking", detail=dim.italic(part.content.strip()))
        case "text" if part.content.strip():
            live.status("Writing")
            answer = markdown(part.content.strip())
            live.stream(bullet("⏺", answer), final=done)
        case "tool-call":
            live.status(f"Calling {part.tool_name}")
        case _:
            pass


def show_tool_result(
    tool_call: ToolCallPart, output: str, live: Live, *, ok: bool
) -> None:
    """Show a tool call, like `⏺ read_file(path="README.md")`, and its output."""
    arguments: dict[str, object] = tool_call.args_as_dict()
    signature = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in arguments.items()
    )
    lines = output.strip().splitlines() or ["(no output)"]
    if len(lines) > _PREVIEW_LINES:
        lines = [*lines[:_PREVIEW_LINES], f"… +{len(lines) - _PREVIEW_LINES} lines"]
    live.print(
        bullet(
            fg("success" if ok else "error", "⏺"),
            bold(tool_call.tool_name) + dim(f"({signature})"),
        ),
        bullet(dim("  ⎿"), fg("muted" if ok else "error", "\n".join(lines))),
    )
