from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on the local environment.
    tqdm = None


def progress(iterable: Iterable[T], **kwargs: object) -> Iterator[T]:
    if tqdm is None:
        return iter(iterable)
    return iter(tqdm(iterable, **kwargs))


def progress_write(message: str) -> None:
    if tqdm is None:
        print(message)
    else:
        tqdm.write(message)
