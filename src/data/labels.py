"""AAMI beat-class mapping shared by every dataset split."""

from typing import Sequence

AAMI_CLASSES = ("N", "S", "V", "F", "Q")

AAMI_SYMBOL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

CLASS_TO_ID = {name: index for index, name in enumerate(AAMI_CLASSES)}


def map_symbol(symbol: str) -> str | None:
    """Map one MIT-BIH annotation symbol to its AAMI class."""
    return AAMI_SYMBOL_MAP.get(symbol)


def encode_aami(labels: Sequence[str]) -> list[int]:
    """Encode labels using one global mapping shared by all splits."""
    unknown = [label for label in labels if label not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"Unknown AAMI labels: {sorted(set(unknown))}")
    return [CLASS_TO_ID[label] for label in labels]
