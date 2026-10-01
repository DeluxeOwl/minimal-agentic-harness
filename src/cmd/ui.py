# Copyright (c) 2026 Andrei Surugiu

from __future__ import annotations

import json
import readline  # ruff: ignore[unused-import]
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Final, Self, final, overload, override

import rich.live
import rich.markdown
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
)
from rich.color import Color, blend_rgb
from rich.color_triplet import ColorTriplet
from rich.console import Console, Group
from rich.rule import Rule
from rich.segment import Segment, SegmentLines
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from agent import ModelFinished, ModelStarted, PartUpdated, ToolFinished, ToolStarted

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from types import TracebackType

    from pydantic_ai import ModelResponsePart, ToolCallPart
    from pydantic_ai.models.openai import OpenAIChatModel
    from rich.console import (
        ConsoleOptions,
        JustifyMethod,
        RenderableType,
        RenderResult,
    )
    from rich.markdown import MarkdownElement

    from agent import AgentEvent

PALETTE: Final[dict[str, tuple[str, str]]] = {
    "accent": ("#a78bfa", "#7c3aed"),
    "muted": ("#a1a1aa", "#52525b"),
    "success": ("#4ade80", "#16a34a"),
    "warning": ("#fbbf24", "#d97706"),
    "error": ("#f87171", "#dc2626"),
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

CODE_THEME: Final = "monokai"

_PROMPT: Final = "❯ "  # ruff: ignore[ambiguous-unicode-character-string]
_PREVIEW_LINES: Final = 3

_FRAMES: Final = "·✢✳✶✻✽✻✶✳✢"
_FRAME_SECONDS: Final = 0.12
_REPAINTS_PER_SECOND: Final = 12.5
_STREAM_SECONDS: Final = 0.05
_SHIMMER_SECONDS: Final = 1.6
_SHIMMER_WIDTH: Final = 3.0
_SHIMMER_GLOW: Final = "#f5f3ff"
_DETAIL_LINES: Final = 3
_BRIGHT: Final = 128
_TERMINAL_BG: Final = ColorTriplet(0x1E, 0x1E, 0x1E)
_FADE: Final = 0.65


def _color(name: str, *, background: bool) -> Color:
    text, back = PALETTE.get(name, (name, name))
    return Color.parse(back if background else text)


@final
class Paint:
    __slots__ = ("_style",)

    def __init__(self, style: Style | None = None) -> None:
        self._style = style or Style()

    def __call__(self, *parts: str | Text) -> Text:
        text = Text.assemble(*parts)
        text.stylize_before(self._style)
        return text

    @override
    def __repr__(self) -> str:
        return f"Paint({str(self._style)!r})"

    @property
    def bold(self) -> Paint:
        return Paint(self._style + Style(bold=True))

    @property
    def dim(self) -> Paint:
        return Paint(self._style + Style(dim=True))

    @property
    def italic(self) -> Paint:
        return Paint(self._style + Style(italic=True))

    @property
    def underline(self) -> Paint:
        return Paint(self._style + Style(underline=True))

    @overload
    def fg(self, color: str, /) -> Paint: ...
    @overload
    def fg(self, color: str, text: str | Text, /, *more: str | Text) -> Text: ...
    def fg(self, color: str, /, *text: str | Text) -> Paint | Text:
        return self._then(Style(color=_color(color, background=False)), text)

    @overload
    def bg(self, color: str, /) -> Paint: ...
    @overload
    def bg(self, color: str, text: str | Text, /, *more: str | Text) -> Text: ...
    def bg(self, color: str, /, *text: str | Text) -> Paint | Text:
        background = _color(color, background=True)
        if self._style.color is None:
            return self._then(
                Style(color=self._readable_on(background), bgcolor=background), text
            )
        return self._then(Style(bgcolor=background), text)

    def blend(self, fade: float = _FADE) -> Paint:
        background = self._style.bgcolor
        if background is None:
            return self
        blended = Color.from_triplet(
            blend_rgb(background.get_truecolor(), _TERMINAL_BG, fade)
        )
        style = Style(bgcolor=blended)
        if self._style.color == self._readable_on(background):
            style += Style(color=self._readable_on(blended))
        return Paint(self._style + style)

    def fill(self, renderable: RenderableType) -> RenderableType:
        return _Fill(renderable, self._style)

    def _then(self, style: Style, text: tuple[str | Text, ...]) -> Paint | Text:
        paint = Paint(self._style + style)
        return paint(*text) if text else paint

    @staticmethod
    def _readable_on(background: Color) -> Color:
        red, green, blue = background.get_truecolor()
        brightness = (299 * red + 587 * green + 114 * blue) / 1000
        return Color.parse("#000000" if brightness > _BRIGHT else "#ffffff")


@final
class _Fill:
    def __init__(self, renderable: RenderableType, style: Style) -> None:
        self._renderable = renderable
        self._style = style

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        for line in console.render_lines(self._renderable, options, pad=False):
            width = sum(segment.cell_length for segment in line)
            slack = options.max_width - width
            yield from line
            if slack > 0:
                yield Segment(" " * slack, self._style)
            yield Segment.line()


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


def echo(*objects: RenderableType) -> None:
    console.print(*objects)


def ask(prompt: str = _PROMPT) -> str | None:
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
            del lines[0]
        marker = [*self._marker.render(console), Segment(" ")]
        blank = [Segment(" " * indent)]
        for index, line in enumerate(lines):
            yield from blank if index else marker
            yield from line
            yield Segment.line()


def bullet(marker: str | Text, content: RenderableType) -> RenderableType:
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
    def __init__(self, *, hint: str = "", tail: int = 6, gap: int = 1) -> None:
        self._status = _Status(hint)
        self._tail = tail
        self._gap = gap
        self._stream: RenderableType | None = None
        self._streamed = 0
        self._lines: list[list[Segment]] = []
        self._rendered_at = 0.0
        self._live = rich.live.Live(
            console=console,
            transient=True,
            refresh_per_second=_REPAINTS_PER_SECOND,
            get_renderable=self._render,
        )

    def __enter__(self) -> Self:
        self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._end_stream()
        self._live.stop()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._status.since

    def status(self, label: str, detail: RenderableType | None = None) -> None:
        if label != self._status.label:
            self._status.label = label
            self._status.since = time.monotonic()
        self._status.detail = detail

    def print(self, *objects: RenderableType) -> None:
        self._end_stream()
        self._space()
        console.print(*objects)

    def stream(self, renderable: RenderableType, *, final: bool = False) -> None:
        if self._stream is None:
            self._space()
        self._stream = renderable
        now = time.monotonic()
        if not final and now - self._rendered_at < _STREAM_SECONDS:
            return
        self._rendered_at = now
        lines = console.render_lines(renderable, pad=False)
        done = len(lines) if final else max(len(lines) - self._tail, self._streamed)
        with console:
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


@final
class AgentRenderer:
    def __init__(self) -> None:
        self._live: Live | None = None
        self._tokens = 0

    @contextmanager
    def turn(self, *, hint: str = "") -> Generator[None]:
        if self._live is not None:
            message = "A renderer turn is already open."
            raise RuntimeError(message)

        started = time.monotonic()
        self._tokens = 0
        try:
            with Live(hint=hint) as live:
                self._live = live
                yield
        except (Exception, KeyboardInterrupt) as error:  # ruff: ignore[blind-except]
            show_error(error)
        else:
            show_turn_summary(time.monotonic() - started, self._tokens)
        finally:
            self._live = None

    def handle(self, event: AgentEvent) -> None:
        live = self._turn_live()
        match event:
            case ModelStarted():
                live.status("Thinking")
            case PartUpdated(part=part, done=done):
                show_response_part(part, live, done=done)
            case ModelFinished(usage=usage):
                self._tokens += usage.output_tokens
            case ToolStarted(tool_call=tool_call):
                live.status(f"Running {tool_call.tool_name}")
            case ToolFinished(tool_call=tool_call, output=output, is_error=is_error):
                show_tool_result(tool_call, output, live, is_error=is_error)

    def preview(self, agent: str, event: AgentEvent) -> None:
        live = self._turn_live()
        detail: Text | None = None
        match event:
            case ModelStarted():
                doing = "Thinking"
            case PartUpdated(part=part) if (
                part.part_kind in {"text", "thinking"} and part.content.strip()
            ):
                doing = "Writing" if part.part_kind == "text" else "Thinking"
                detail = _one_line(dim.italic(part.content.strip().splitlines()[-1]))
            case ToolStarted(tool_call=tool_call):
                doing = f"Running {tool_call.tool_name}"
                detail = _one_line(
                    dim(f"{tool_call.tool_name}({_signature(tool_call)})")
                )
            case ModelFinished(usage=usage):
                self._tokens += usage.output_tokens
                return
            case _:
                return
        live.status(f"{agent} › {doing}", detail=detail)  # ruff: ignore[ambiguous-unicode-character-string]

    def _turn_live(self) -> Live:
        if self._live is None:
            message = "Open a renderer turn before handling agent events."
            raise RuntimeError(message)
        return self._live


def show_model_info(model: OpenAIChatModel) -> None:
    echo(
        bold.bg("accent", " ✻ minimal-agentic-harness "),
        dim(f"{model.model_name} at {model.base_url}"),
    )


def show_tools_info(names: Iterable[str]) -> None:
    echo(dim(f"tools: {', '.join(names)} · /system to see the prompt · ctrl+d to quit"))
    echo()


def show_block(header: str, body: str, *, color: str = "accent") -> None:
    block = Syntax(
        f"{header}\n\n{body}",
        "text",
        theme=CODE_THEME,
        word_wrap=True,
        padding=1,
    )
    block.stylize_range(
        Style(bold=True, color=_color(color, background=False)),
        (1, 0),
        (1, len(header)),
    )
    echo()
    echo(block)


def show_skill_loaded(name: str) -> None:
    echo(bold.bg("purple", f" [skill] {name} loaded "))


def show_system_prompt(prompt: str) -> None:
    tokens = len(prompt) // 4
    show_block(
        f"system prompt · {len(prompt)} chars · ~{tokens} tokens",
        prompt,
        color="accent",
    )


def _explain(error: BaseException) -> str:
    match error:
        case KeyboardInterrupt():
            return "Interrupted"
        case ModelHTTPError() if "exceed_context_size_error" in str(error.body):
            return "The conversation is too long for the model."
        case ModelHTTPError():
            return f"The model server failed ({error.status_code}): {error.body}"
        case ModelAPIError():
            return "No answer from remote url. Is `make serve-llm` running?"
        case _:
            return f"The model replied with something unexpected: {error}"


def show_error(error: BaseException) -> None:
    echo(fg("error", f"  ⎿ {_explain(error)}"))


def show_turn_summary(seconds: float, tokens: int) -> None:
    speed = f"{tokens / seconds:.0f} tok/s"
    echo()
    echo(dim(f"✻ Worked for {seconds:.1f}s · {tokens} tokens · {speed}"))


def show_response_part(
    part: ModelResponsePart, live: Live, *, done: bool = False
) -> None:
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


def _signature(tool_call: ToolCallPart) -> str:
    arguments: dict[str, object] = tool_call.args_as_dict()
    return ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in arguments.items()
    )


def _one_line(text: Text) -> Text:
    text.no_wrap = True
    text.overflow = "ellipsis"
    return text


def show_tool_result(
    tool_call: ToolCallPart, output: str, live: Live, *, is_error: bool
) -> None:
    lines = output.strip().splitlines() or ["(no output)"]
    if len(lines) > _PREVIEW_LINES:
        lines = [*lines[:_PREVIEW_LINES], f"… +{len(lines) - _PREVIEW_LINES} lines"]
    live.print(
        bullet(
            fg("error" if is_error else "success", "⏺"),
            bold(tool_call.tool_name) + dim(f"({_signature(tool_call)})"),
        ),
        bullet(dim("  ⎿"), fg("error" if is_error else "muted", "\n".join(lines))),
    )
