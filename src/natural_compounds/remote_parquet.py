"""Read single parquet row groups from a remote file over HTTP range requests.

A ZINC shard is 177 MB but holds 1,955 row groups of a thousand rows. Pulling one
row group costs the footer plus about 0.3 MB, so a sample spread over many shards
is cheap. Downloading whole shards instead is what produced the tranche-blocked
pool this replaces.
"""

from __future__ import annotations

import io
import urllib.request


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges. Enough for pyarrow's reader."""

    def __init__(self, url: str, timeout: int = 120):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        self.bytes_read = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            self.length = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.length + offset
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.pos
        if size == 0 or self.pos >= self.length:
            return b""
        end = min(self.pos + size, self.length) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data
