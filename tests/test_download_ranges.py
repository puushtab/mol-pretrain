"""Byte-range partitioning for the parallel downloader."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
)
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


def test_ranges_cover_every_byte_once():
    spans = download_data.ranges(1000, 8)
    assert spans[0][0] == 0 and spans[-1][1] == 999
    assert all(b[0] == a[1] + 1 for a, b in zip(spans, spans[1:], strict=False))
    assert sum(e - s + 1 for s, e in spans) == 1000


def test_small_file_uses_fewer_parts():
    assert download_data.ranges(3, 8) == [(0, 0), (1, 1), (2, 2)]


def test_single_part_when_asked():
    assert download_data.ranges(1000, 1) == [(0, 999)]
