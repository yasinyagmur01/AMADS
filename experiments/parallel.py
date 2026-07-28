"""Shared helper: run independent experiment reps with bounded concurrency."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def run_specs_parallel(
    specs: list,
    run_one: Callable[..., Awaitable[T]],
    *,
    concurrency: int = 3,
    run_kwargs: dict | None = None,
) -> tuple[list[T], float]:
    """
    Execute independent run specs concurrently (rounds within a run stay sequential
    inside run_one). Returns (results_in_spec_order, wall_clock_seconds).
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    kwargs = run_kwargs or {}
    results: list[T | None] = [None] * len(specs)

    async def _wrapped(idx: int, spec) -> None:
        async with sem:
            results[idx] = await run_one(spec, **kwargs)

    t0 = time.perf_counter()
    await asyncio.gather(*[_wrapped(i, s) for i, s in enumerate(specs)])
    elapsed = time.perf_counter() - t0
    return [r for r in results if r is not None], elapsed  # type: ignore[misc]
