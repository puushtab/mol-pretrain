"""The diagram of `docs/remaining-steps.md`: what is done, and the steps that remain.

    uv run python scripts/plot_remaining_steps.py

Reads `docs/figures/remaining-steps.json` and writes `remaining-steps.png` and
`remaining-steps.svg` beside it. The top band is one card per finished stage:
what ran, what it showed, and how we read it. Below it the steps that remain are
drawn as lanes over a date axis.

Nothing is computed here. Every number on the figure is typed into the JSON
file and has to be copied from the tables under `docs/results/`; change the
plan or a result there, then run this again.

The JSON file holds `title`, `subtitle`, and:

- `done`: `{"heading", "cards": [{"head", "when", "body", "reading",
  "highlight" (optional)}]}`.
- `todo`: `{"heading", "note", "first", "last", "lanes": [[key, label]],
  "milestones": [{"date", "label", "ha"}], "steps": [{"id", "title", "lane",
  "start", "end", "kind": "due" | "critical" | "slack" | "optional", "rows",
  "row", "label_days", "arrows_from": [...]}], "due_label"}`. Dates are ISO
  days; a step covers start to end inclusive. `rows` splits a lane into that
  many rows and `row` picks one. `label_days` writes the name to the right of a
  box too narrow to hold it, over that many days. `arrow_at` (0 to 1) moves a
  vertical arrow sideways within the days the two steps share. `due` marks what the nearest
  deadline needs, and `due_label` is its legend entry.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.textpath import TextPath

SPEC = Path(__file__).resolve().parents[1] / "docs" / "figures" / "remaining-steps.json"

BLUE, PALE, EDGE = "#2a78d6", "#cfe2fb", "#7fb0ee"
INK, SOFT, FAINT = "#0b0b0b", "#52514e", "#9a9992"
GRID, BAND, PAPER, CARD = "#e1e0d9", "#f6f5f1", "#fcfcfa", "#ffffff"
DUE, DUE_FILL = "#c2570c", "#fbe4d0"  # what the nearest deadline needs

WIDTH = 15.5  # inches
SIDE = 0.2  # left and right margin of the page
LANE_LABELS = 1.3  # width kept for the lane names
LANE = 1.5  # height of one lane
CARD_HEIGHT = 2.8
GAP = 0.14  # between two cards


def wrap(text: str, width: float, size: float, weight: str = "normal") -> str:
    """Break `text` so that no line is wider than `width` inches at this font size."""
    font = FontProperties(weight=weight)

    def inches(line: str) -> float:
        return TextPath((0, 0), line, size=size, prop=font).get_extents().width / 72

    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            longer = f"{line} {word}".strip()
            if line and inches(longer) > width:
                lines.append(line)
                line = word
            else:
                line = longer
        lines.append(line)
    return "\n".join(lines)


def draw_done(ax, done: dict) -> None:
    """One card per finished stage, left to right in the order they ran."""
    cards = done["cards"]
    span = WIDTH - 2 * SIDE
    width = (span - GAP * (len(cards) - 1)) / len(cards)
    inner = width - 0.3
    ax.set_xlim(0, span)
    ax.set_ylim(0, CARD_HEIGHT)
    ax.axis("off")
    for i, card in enumerate(cards):
        x0 = i * (width + GAP)
        strong = card.get("highlight", False)
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.02),
                width,
                CARD_HEIGHT - 0.04,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=CARD,
                edgecolor=BLUE if strong else GRID,
                linewidth=1.8 if strong else 1.2,
            )
        )
        left = x0 + 0.15
        ax.text(
            left, CARD_HEIGHT - 0.16, card["head"], fontsize=10.6, fontweight="bold",
            color=INK, va="top",
        )  # fmt: skip
        when = wrap(card["when"], inner, 8.2)
        ax.text(left, CARD_HEIGHT - 0.4, when, fontsize=8.2, color=SOFT, va="top")
        body_at = CARD_HEIGHT - 0.54 - 0.15 * (when.count("\n") + 1)
        ax.text(
            left, body_at, wrap(card["body"], inner, 8.6), fontsize=8.6, color=INK, va="top",
            linespacing=1.32,
        )  # fmt: skip
        reading = wrap(card["reading"], inner, 8.9, "bold")
        rule = 0.2 + 0.172 * (reading.count("\n") + 1)
        ax.plot([left, x0 + width - 0.15], [rule, rule], color=GRID, linewidth=1)
        ax.text(
            left, 0.13, reading, fontsize=8.9, fontweight="bold", color=BLUE, va="bottom",
            linespacing=1.3,
        )  # fmt: skip


def draw_todo(ax, todo: dict, inches_wide: float) -> None:
    """The steps that remain, as lanes over a date axis."""
    first, last = date.fromisoformat(todo["first"]), date.fromisoformat(todo["last"])
    lanes = todo["lanes"]
    lane_y = {key: len(lanes) - 1 - i for i, (key, _) in enumerate(lanes)}

    def x(day: str | date) -> float:
        day = date.fromisoformat(day) if isinstance(day, str) else day
        return (day - first).days

    days = x(last) + 1
    per_day = inches_wide / days

    for key, label in lanes:
        y = lane_y[key]
        if (len(lanes) - 1 - y) % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0)
        ax.text(
            -0.25, y, label, ha="right", va="center", fontsize=10.5, fontweight="bold", color=INK
        )

    day = first
    while day <= last:
        ax.axvline(x(day), color=GRID, linewidth=0.7, zorder=1)
        colour = FAINT if day.weekday() >= 5 else SOFT
        ax.text(
            x(day) + 0.5, -0.66, f"{day.day}", ha="center", va="top", fontsize=8.6, color=colour
        )
        ax.text(
            x(day) + 0.5, -0.86, "MTWTFSS"[day.weekday()], ha="center", va="top", fontsize=7,
            color=colour,
        )  # fmt: skip
        if day.day == 1 or day == first:
            ax.text(
                x(day) + 0.05, -1.1, day.strftime("%B"), ha="left", va="top", fontsize=8.6,
                color=SOFT, fontweight="bold",
            )  # fmt: skip
        day += timedelta(days=1)
    ax.axvline(days, color=GRID, linewidth=0.7, zorder=1)

    top = len(lanes) - 0.5
    for stone in todo["milestones"]:
        at = x(stone["date"]) + 1.0
        ax.plot(  # stops above the dates, so that it does not cross the month names
            [at, at], [-0.5, top + 0.42], color=INK, linewidth=1.3, linestyle=(0, (1, 2)), zorder=2
        )
        side = stone.get("ha", "right")
        ax.text(
            at + (-0.1 if side == "right" else 0.1), top + 0.08, stone["label"], ha=side,
            va="bottom", fontsize=9.2, fontweight="bold", color=INK,
        )  # fmt: skip

    styles = {
        "critical": dict(facecolor=BLUE, edgecolor=BLUE, linewidth=1.5),
        "slack": dict(facecolor=PALE, edgecolor=EDGE, linewidth=1.5),
        "due": dict(facecolor=DUE_FILL, edgecolor=DUE, linewidth=1.8),
        "optional": dict(facecolor=PAPER, edgecolor=BLUE, linewidth=1.5, linestyle=(0, (4, 3))),
    }
    boxes = {}
    for step in todo["steps"]:
        rows, row = step.get("rows", 1), step.get("row", 0)
        height = 0.78 / rows
        y0 = lane_y[step["lane"]] + 0.39 - (row + 1) * height + 0.03 * (rows - 1)
        x0, x1 = x(step["start"]) + 0.08, x(step["end"]) + 0.92
        kind = step["kind"]
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0 + 0.02),
                x1 - x0,
                height - 0.07,
                boxstyle="round,pad=0,rounding_size=0.09",
                zorder=3,
                **styles[kind],
            )
        )
        weight = "bold" if kind == "critical" else "normal"
        size = step.get("fontsize", 8.4)
        text = f"{step['id']}: {step['title']}"
        if "label_days" in step:  # a box too narrow for its name: write it to the right
            ax.text(
                x1 + 0.2, y0 + height / 2 - 0.015,
                wrap(text, step["label_days"] * per_day - 0.1, size, "bold"), ha="left",
                va="center", fontsize=size, linespacing=1.25, fontweight="bold", color=INK,
                zorder=4,
            )  # fmt: skip
        else:
            ax.text(
                (x0 + x1) / 2, y0 + height / 2 - 0.015,
                wrap(text, (x1 - x0) * per_day - 0.16, size, weight), ha="center", va="center",
                fontsize=size, linespacing=1.25, fontweight=weight, zorder=4,
                color="white" if kind == "critical" else INK,
            )  # fmt: skip
        boxes[step["id"]] = (x0, x1, y0 + 0.02, y0 + height - 0.05)

    for step in todo["steps"]:
        for source in step.get("arrows_from", []):
            sx0, sx1, sy0, sy1 = boxes[source]
            tx0, tx1, ty0, ty1 = boxes[step["id"]]
            start, end = (sx1, (sy0 + sy1) / 2), (tx0, (ty0 + ty1) / 2)
            if tx0 < sx1 + 0.05:  # the target starts under or above the source: go vertically
                low, high = max(sx0, tx0), min(sx1, tx1)
                mid = low + step.get("arrow_at", 0.5) * (high - low)
                up = ty0 > sy1
                start, end = (mid, sy1 if up else sy0), (mid, ty0 if up else ty1)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=SOFT,
                    linewidth=1.3,
                    zorder=5,
                    shrinkA=2,
                    shrinkB=2,
                )
            )

    ax.set_xlim(-0.02, days + 0.02)
    ax.set_ylim(-1.3, top + 0.42)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    done, todo = spec["done"], spec["todo"]

    plt.rcParams["svg.hashsalt"] = "remaining-steps"
    lanes_height = LANE * len(todo["lanes"]) + 1.5
    note_lines = todo["note"].count("\n") + 1
    # From the top of the page down, in inches.
    title_at, subtitle_at, done_heading_at = 0.3, 0.68, 1.22
    cards_top = done_heading_at + 0.32
    todo_heading_at = cards_top + CARD_HEIGHT + 0.42
    lanes_top = todo_heading_at + 0.3 + 0.2 * note_lines + 0.1
    height = lanes_top + lanes_height + 0.5

    fig = plt.figure(figsize=(WIDTH, height))
    fig.patch.set_facecolor(PAPER)

    def up(inches_from_top: float) -> float:
        return 1 - inches_from_top / height

    fig.text(
        SIDE / WIDTH, up(title_at), spec["title"], fontsize=15.5, fontweight="bold", color=INK,
        va="top",
    )  # fmt: skip
    fig.text(
        SIDE / WIDTH, up(subtitle_at), spec["subtitle"], fontsize=9.6, color=SOFT, va="top",
        linespacing=1.45,
    )  # fmt: skip

    fig.text(
        SIDE / WIDTH, up(done_heading_at), done["heading"], fontsize=11.5, fontweight="bold",
        color=INK, va="top",
    )  # fmt: skip
    cards = fig.add_axes(
        (SIDE / WIDTH, up(cards_top + CARD_HEIGHT), 1 - 2 * SIDE / WIDTH, CARD_HEIGHT / height)
    )
    draw_done(cards, done)

    fig.text(
        SIDE / WIDTH, up(todo_heading_at), todo["heading"], fontsize=11.5, fontweight="bold",
        color=INK, va="top",
    )  # fmt: skip
    fig.text(
        SIDE / WIDTH, up(todo_heading_at + 0.3), todo["note"], fontsize=9.6, color=SOFT, va="top",
        linespacing=1.45,
    )  # fmt: skip
    left = SIDE + LANE_LABELS
    lanes = fig.add_axes(
        (
            left / WIDTH,
            up(lanes_top + lanes_height),
            (WIDTH - left - SIDE) / WIDTH,
            lanes_height / height,
        )
    )
    lanes.set_facecolor(PAPER)
    draw_todo(lanes, todo, WIDTH - left - SIDE)

    fig.legend(
        handles=[
            Patch(facecolor=DUE_FILL, edgecolor=DUE, label=todo["due_label"]),
            Patch(facecolor=BLUE, edgecolor=BLUE, label="sets the finishing date"),
            Patch(facecolor=PALE, edgecolor=EDGE, label="has slack"),
            Patch(
                facecolor=PAPER, edgecolor=BLUE, linestyle=(0, (4, 3)),
                label="optional: only if agreed and written down first",
            ),
        ],
        loc="lower left", bbox_to_anchor=(SIDE / WIDTH - 0.004, 0.006), ncol=4, frameon=False,
        fontsize=9.2, labelcolor=SOFT, handlelength=2.2, columnspacing=2.2,
    )  # fmt: skip

    for suffix in ("png", "svg"):
        out = args.spec.with_suffix(f".{suffix}")
        extra = {"metadata": {"Date": None}} if suffix == "svg" else {}
        fig.savefig(out, dpi=170, facecolor=PAPER, **extra)
        if suffix == "svg":  # matplotlib leaves trailing blanks, which the commit hook rewrites
            out.write_text("\n".join(line.rstrip() for line in out.read_text().splitlines()) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
