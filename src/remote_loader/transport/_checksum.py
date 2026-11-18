"""Required zlib-ng CRC32 backend, wire-compatible with stdlib zlib."""

try:
    from zlib_ng.zlib_ng import crc32 as crc32
except (ImportError, OSError) as exc:
    raise ImportError(
        "Failed to load required CRC32 backend 'zlib-ng'. "
        "Install or reinstall a zlib-ng build compatible with this Python version and platform."
    ) from exc
