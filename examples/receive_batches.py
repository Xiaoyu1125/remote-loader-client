"""Inspect server-delivered VLA canonical samples or complete VQA v2 batches.

VLA images retain the requested raw uint8 or server-processed float32 values.
This example only summarizes the delivered schema; it does not preprocess pixels.
"""

from remote_loader import RemoteDataLoader


def describe(value):
    if isinstance(value, dict):
        return {name: describe(child) for name, child in value.items()}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": tuple(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__}


def receive(spec_path: str) -> None:
    with RemoteDataLoader.from_yaml(spec_path) as loader:
        for item in loader:
            print(describe(item))
