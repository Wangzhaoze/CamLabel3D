from __future__ import annotations

import numpy as np
import pytest

from camlabel3d.core.frame_cache import FrameCache


def _frame(value: int, byte_count: int = 4) -> np.ndarray:
    return np.full((byte_count,), value, dtype=np.uint8)


def test_recent_access_controls_lru_eviction_and_stats() -> None:
    cache = FrameCache(max_bytes=8)
    cache.put(1, _frame(1))
    cache.put(2, _frame(2))

    assert cache.get(1) is not None  # Make frame 1 newer than frame 2.
    cache.put(3, _frame(3))

    assert cache.get(2) is None
    assert cache.get(1) is not None
    assert cache.get(3) is not None
    stats = cache.stats()
    assert stats.entries == 2
    assert stats.bytes_used == 8
    assert stats.max_bytes == 8
    assert stats.hits == 3
    assert stats.misses == 1


def test_cached_frames_are_contiguous_and_read_only() -> None:
    source = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    assert not source.flags.c_contiguous
    cache = FrameCache(max_bytes=source.nbytes)

    cached = cache.put(7, source)

    assert cached.flags.c_contiguous
    assert cached.flags.writeable is False
    assert source.flags.writeable is True
    assert cache.get(7) is cached
    with pytest.raises(ValueError):
        cached[0, 0] = 255


def test_memory_limit_is_never_exceeded_and_oversized_frames_are_not_cached() -> None:
    cache = FrameCache(max_bytes=6)

    oversized = cache.put(10, _frame(10, byte_count=7))

    assert oversized.flags.writeable is False
    assert cache.get(10) is None
    assert cache.stats().entries == 0
    assert cache.stats().bytes_used == 0

    cache.put(1, _frame(1, byte_count=4))
    cache.put(2, _frame(2, byte_count=4))
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.bytes_used == 4
    assert stats.bytes_used <= stats.max_bytes


def test_replacing_an_entry_updates_accounted_bytes_and_clear_resets_storage() -> None:
    cache = FrameCache(max_bytes=10)
    cache.put(1, _frame(1, byte_count=3))
    cache.put(1, _frame(2, byte_count=6))

    assert cache.stats().entries == 1
    assert cache.stats().bytes_used == 6

    cache.clear()

    assert cache.stats().entries == 0
    assert cache.stats().bytes_used == 0
