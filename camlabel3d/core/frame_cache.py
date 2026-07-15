"""Small, thread-safe, memory-bounded cache for decoded RGB frames."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameCacheStats:
    entries: int
    bytes_used: int
    max_bytes: int
    hits: int
    misses: int


class FrameCache:
    """LRU cache bounded by decoded byte size rather than frame count."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes_used = 0
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def get(self, index: int) -> np.ndarray | None:
        key = int(index)
        with self._lock:
            frame = self._entries.get(key)
            if frame is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return frame

    def put(self, index: int, frame: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(frame)
        value.setflags(write=False)
        key = int(index)
        if self.max_bytes <= 0 or value.nbytes > self.max_bytes:
            return value

        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes_used -= int(previous.nbytes)
            self._entries[key] = value
            self._bytes_used += int(value.nbytes)
            while self._bytes_used > self.max_bytes and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes_used -= int(evicted.nbytes)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes_used = 0

    def stats(self) -> FrameCacheStats:
        with self._lock:
            return FrameCacheStats(
                entries=len(self._entries),
                bytes_used=self._bytes_used,
                max_bytes=self.max_bytes,
                hits=self._hits,
                misses=self._misses,
            )
