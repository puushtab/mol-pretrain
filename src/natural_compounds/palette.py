"""One colour per corpus, and one order, for every figure and every table.

The notebook defined these and the three scripts that write PNGs did not, so a
reader moving from a notebook figure to a saved one saw COCONUT change from
green to orange and ZINC from grey to blue. Worse than ugly: the two palettes
swapped a pair, so the same colour meant different corpora on facing pages.

Natural-product-like first, general libraries last, so a table and a figure read
in the same direction.
"""

from __future__ import annotations

ORDER = ["coconut", "zinc_nplike", "pubchem", "zinc_general"]

COLOUR = {
    "coconut": "green",
    "zinc_nplike": "orange",
    "pubchem": "blue",
    "zinc_general": "grey",
    # Benchmarks are drawn over the corpora, so they need two colours no corpus
    # uses. They are not corpora and the shape says so too.
    "task:bbbp": "purple",
    "task:esol": "red",
}

LABEL = {
    "coconut": "COCONUT",
    "zinc_nplike": "ZINC, NP-like decile",
    "pubchem": "PubChem (CheMeleon's)",
    "zinc_general": "ZINC, random",
}


def ordered(names) -> list[str]:
    """The names given, in the project's order, with anything unknown last."""
    known = [n for n in ORDER if n in names]
    return known + [n for n in names if n not in ORDER]


def colour(name: str) -> str:
    return COLOUR.get(name, "black")
