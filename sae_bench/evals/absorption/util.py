from collections.abc import Generator, Sequence
from typing import TypeVar, overload

import torch
from tqdm.autonotebook import tqdm

T = TypeVar("T")
K = TypeVar("K")


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@overload
def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]: ...


@overload
def batchify(
    data: torch.Tensor, batch_size: int, show_progress: bool = False
) -> Generator[torch.Tensor, None, None]: ...


def batchify(
    data: Sequence[T] | torch.Tensor, batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T] | torch.Tensor, None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for start_idx, end_idx in batchify_indices(len(data), batch_size, show_progress):
        yield data[start_idx:end_idx]


def batchify_indices(
    total_items: int, batch_size: int, show_progress: bool = False
) -> Generator[tuple[int, int], None, None]:
    """Generate batches of indices from 0 to total_items."""
    for i in tqdm(
        range(0, total_items, batch_size),
        total=(total_items // batch_size + (total_items % batch_size != 0)),
        disable=not show_progress,
    ):
        yield i, i + batch_size


def flip_dict(d: dict[T, T]) -> dict[T, T]:
    """Flip a dictionary, i.e. {a: b} -> {b: a}"""
    return {v: k for k, v in d.items()}


def listify(item: T | list[T]) -> list[T]:
    """Convert an item or list of items to a list."""
    if isinstance(item, list):
        return item
    return [item]


def dict_zip(*dicts: dict[T, K]) -> Generator[tuple[T, tuple[K, ...]], None, None]:
    """Zip together multiple dictionaries, iterating their common keys and a tuple of values."""
    if not dicts:
        return
    keys = set(dicts[0]).intersection(*dicts[1:])
    for key in keys:
        yield key, tuple(d[key] for d in dicts)
